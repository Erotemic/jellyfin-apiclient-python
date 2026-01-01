"""
Async Jellyfin media graph crawler with "filesystem-like" navigation.

Progress semantics (subtree-completion model, DAG-safe):
- Persistent top-level bar: one unit per top-level media folder; advances ONLY when that folder's
  entire subtree (within max_depth) is finished.
- Transient per-top-level-folder bar: total = TotalRecordCount for that folder; advances by 1
  for each DIRECT child in that listing when that direct child is "complete":
    * leaf child => complete immediately when discovered
    * folder child => complete when its subtree is finished IF this parent is the child's "primary"
      parent; otherwise complete immediately (prevents hangs in DAG situations)
  Removed when the top-level folder completes.
- Optional transient "large folder" bars for subfolders with many direct children (>= threshold):
  same semantics; removed when folder completes.

Important invariants:
- The NetworkX DiGraph is used.
- The rich label strings and how they are computed are unchanged (see _update_graph_labels).
- Graph construction details may vary, but "browse" is supported: setup defaults to only listing
  top-level folders; user can expand each individually via open_node/cd.
"""

import typing
import rich
import ubelt as ub
import networkx as nx


class MediaGraph:
    """
    Wraps a Jellyfin API client with an interface to walk media folders.

    Builds a graph of all media items in a jellyfin database. A current working
    directory pointer is maintained to allow filesystem like navigation of the
    database.

    Maintains an in-memory graph of the jellyfin database state. This allows
    for efficient client-side queries and exploration, but does take some time
    to construct, and is not kept in sync with the server in the case of
    server-side changes.

    Example:
        >>> from jellyfin_apiclient_python.media_graph import MediaGraph
        >>> import ubelt as ub
        >>> # Given an API client
        >>> #MediaGraph.ensure_demo_server(reset=0)
        >>> client = MediaGraph.demo_client()
        >>> client._authed = client._authed.with_timeout(90)
        >>> # Create the media graph by passing it the client
        >>> from jellyfin_apiclient_python.media_graph import MediaGraph
        >>> self = MediaGraph(client)
        >>> self.walk_config['initial_depth'] = None
        >>> self.walk_config['perquery_limit'] = 100
        >>> self._DEBUG = 0
        >>> self.setup()
        ...
        >>> # Print the graph at the top level
        >>> self.tree()
        ╟──  f137a2dd21bbc1b99aa5c0f6bf02a805 : 📂 CollectionFolder - Movies
        ╎   ├─╼  826931c9344d013db9db13341db65cce : 🎥 Movie - The great train robbery
        ╎   ├─╼  6144770939e7eeef8d9bd4eb519bf770 : 🎥 Movie - Popeye the Sailor meets Sinbad the Sailor
        ╎   ├─╼  0a8c358081cc4bf1eb74a660ca8616f4 : 🎥 Movie - File:Zur%C3%BCck_in_die_Zukunft_(Film)_01
        ╎   └─╼  32a52b6711776ffeb09b0e737aab5558 : 🎥 Movie - Popeye the Sailor meets Sinbad the Sailor
        ╟──  7e64e319657a9516ec78490da03edccb : 📂 CollectionFolder - Music
        ╎   ├─╼  8288fbf650ae583fc36d715b2c82dff5 : ♬ Audio - Zurück in die Zukunft
        ╎   ├─╼  76ed290f795e4a24a9cceba4aa8bfb33 : ♬ Audio - Heart_Monitor_Beep--freesound.org
        ╎   └─╼  7a7f9d14d80062884dbefd156818b339 : ♬ Audio - Clair De Lune
        ╙──  1071671e7bffa0532e930debee501d2e : 📂 ManualPlaylistsFolder - Playlists
        >>> # Search for items based on name
        >>> found = list(self.find('the'))
        >>> print(f'found = {ub.urepr(found, nl=1)}')
        found = [
            '6144770939e7eeef8d9bd4eb519bf770',
            '32a52b6711776ffeb09b0e737aab5558',
        ]
        >>> # List the folder nodes at the top level
        >>> top_level = self.ls()
        >>> print(f'top_level = {ub.urepr(top_level, nl=1)}')
        top_level = [
            'f137a2dd21bbc1b99aa5c0f6bf02a805',
            '7e64e319657a9516ec78490da03edccb',
            '1071671e7bffa0532e930debee501d2e',
        ]
        >>> # Change the CWD to the music folder
        >>> self.cd('7e64e319657a9516ec78490da03edccb')
        >>> # Print the graph at the CWD
        >>> self.tree()
        ╙──  7e64e319657a9516ec78490da03edccb : 📂 CollectionFolder - Music
            ├─╼  8288fbf650ae583fc36d715b2c82dff5 : ♬ Audio - Zurück in die Zukunft
            ├─╼  76ed290f795e4a24a9cceba4aa8bfb33 : ♬ Audio - Heart_Monitor_Beep--freesound.org
            └─╼  7a7f9d14d80062884dbefd156818b339 : ♬ Audio - Clair De Lune
        >>> # Searching is in the context of the cwd
        >>> found = list(self.find('the'))
        >>> print(f'found = {ub.urepr(found, nl=1)}')
        []
        >>> found = list(self.find('Clair'))
        >>> print(f'found = {ub.urepr(found, nl=1)}')
        found = [
            '7a7f9d14d80062884dbefd156818b339',
        ]
        >>> # Print details about a specific item
        >>> self.print_item('7a7f9d14d80062884dbefd156818b339')
        node=7a7f9d14d80062884dbefd156818b339
        properties = {
            'expanded': False,
        }
        item = {
            'Name': 'Clair De Lune',
            ...
            'Id': '7a7f9d14d80062884dbefd156818b339',
            ...
            'Path': '/media/music/Clair_de_Lune_-_Wright_Brass_-_United_States_Air_Force_Band_of_Flight.mp3',
            ...
            'MediaType': 'Audio',
        }
    """
    def __init__(self, client):
        self.client = client
        self.graph = None
        self.walk_config = {
            # Browse mode default: only initialize top-level roots on setup.
            # None => crawl everything recursively.
            'initial_depth': 0,
            'include_collection_types': None,
            'exclude_collection_types': None,
            'perquery_limit': 100,
            'query_attempts': 1,

            # Async / concurrency knobs
            'max_concurrent_requests': 20,
            'max_concurrent_parents': 10,
            'max_concurrent_root_walks': 3,
            'page_prefetch': True,

            # Progress behavior
            'info_update_interval': 0.5,
            'large_folder_threshold': 500,

            # Debug
            'trace': False,
        }
        self.display_config = {
            'show_path': False,
        }
        self._cwd = None
        self._cwd_children = None
        self._media_root_nodes = None
        self._DEBUG = False

        # In-flight expansion de-dup across concurrent expansions of the same node.
        self._inflight = set()
        self._inflight_lock = None  # asyncio.Lock, initialized lazily

        from jellyfin_apiclient_python.openapi._generated.models.item_fields import ItemFields
        self.fields = [ItemFields.PATH, ItemFields.GENRES, ItemFields.PARENTID]

    def _dbg(self, msg: str):
        if self._DEBUG:
            print(f'[MediaGraph] {msg}')

    def _trace(self, event: str, **kw):
        if not self.walk_config.get('trace', False):
            return
        rich.print(f"[dim]{event}[/dim] {ub.urepr(kw, nl=0)}")

    @classmethod
    def _run_async(cls, coro):
        """Run an async coroutine, creating/managing an event loop if needed."""
        import asyncio
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        import threading
        result_box = {}
        error_box = {}

        def _thread_main():
            try:
                result_box['result'] = asyncio.run(coro)
            except Exception as ex:
                error_box['error'] = ex

        t = threading.Thread(target=_thread_main, daemon=True)
        t.start()
        t.join()
        if 'error' in error_box:
            raise error_box['error']
        return result_box.get('result', None)

    @classmethod
    def ensure_demo_server(cls, reset: bool = False):
        from jellyfin_apiclient_python.demo import JellyfinDockerServer
        server = JellyfinDockerServer(reuse_container=not reset)
        server.start()
        cls._demo_server = server
        return server

    @classmethod
    def demo_client(cls):
        from jellyfin_apiclient_python.openapi.client import Jellyfin
        url = 'http://127.0.1.1:34907'
        username = 'jellyfin-user'
        password = 'jellyfin-pass'
        client = Jellyfin(base_url=url, username=username, password=password)
        client.login()
        return client

    # -------------------------
    # Navigation / UI
    # -------------------------

    def tree(self, max_depth=None):
        if self._cwd is None:
            self.print_graph(max_depth=max_depth)
        else:
            self.print_graph(sources=[self._cwd], max_depth=max_depth)

    def ls(self):
        if self._cwd is None:
            return self._media_root_nodes
        else:
            return self._cwd_children

    def cd(self, node):
        self._cwd = node
        if node is None:
            self._cwd_children = self._media_root_nodes
        else:
            if not self.graph.nodes[node]['item'].get('IsFolder', False):
                raise Exception('can only cd into a folder')
            self.open_node(node, verbose=0)
            self._cwd_children = list(self.graph.succ[node])

    def __truediv__(self, node):
        self.open_node(node, verbose=1)
        return self

    # -------------------------
    # Public setup / expansion
    # -------------------------

    def setup(self):
        self._run_async(self.setup_async())
        self._update_graph_labels()

    async def setup_async(self):
        await self._init_media_folders_async()
        return self

    def open_node(self, node, verbose=0, max_depth=1):
        if self.graph is None:
            raise RuntimeError('MediaGraph.graph is not initialized; call setup() first')

        if isinstance(node, str):
            node_id = node
            node_data = self.graph.nodes[node_id]
        elif isinstance(node, dict) and 'item' in node:
            node_data = node
            node_id = node_data['item']['Id']
        else:
            node_id = str(node)
            node_data = self.graph.nodes[node_id]

        item = node_data['item']

        stats = {
            'node_types': ub.ddict(int),
            'edge_types': ub.ddict(int),
            'nondag_edge_types': ub.ddict(int),
            'total': 0,
            'latest_name': None,
            'latest_path': None,
        }

        pman = _RichWalkProgress(enabled=True)
        with pman:
            self._run_async(self._walk_node_async([item], pman, stats, max_depth=max_depth, root_bar=False))
        self._update_graph_labels(sources=[node_id])

        if verbose:
            self.print_graph([node])
            self.print_item(node)
        return self

    # -------------------------
    # Initialization
    # -------------------------

    async def _init_media_folders_async(self):
        """Initialize graph with top-level media folders and optionally pre-walk them."""
        client = self.client
        graph = nx.DiGraph()
        self.graph = graph

        include_collection_types = self.walk_config.get('include_collection_types', None)
        exclude_collection_types = self.walk_config.get('exclude_collection_types', None)
        initial_depth = self.walk_config['initial_depth']

        resp = await client.api.library.get_media_folders.asyncio_detailed()
        assert resp.status_code == 200
        data = resp.parsed.to_dict()

        root_items = []
        root_node_ids = []
        stats = {
            'node_types': ub.ddict(int),
            'edge_types': ub.ddict(int),
            'nondag_edge_types': ub.ddict(int),
            'total': 0,
            'latest_name': None,
            'latest_path': None,
        }

        for folder in data.get('Items', []):
            candidates = folder.get('Children') or [folder]
            for item in candidates:
                collection_type = item.get('CollectionType', folder.get('CollectionType', None))
                if include_collection_types is not None and collection_type not in include_collection_types:
                    continue
                if exclude_collection_types is not None and collection_type in exclude_collection_types:
                    continue

                item_id = item['Id']
                item['type'] = item.get('Type', item.get('type', None))
                if item_id not in graph:
                    graph.add_node(item_id, item=item, properties=dict(expanded=False))

                root_items.append(item)
                root_node_ids.append(item_id)

        self._media_root_nodes = root_node_ids

        pman = _RichWalkProgress(enabled=True)
        with pman:
            import asyncio
            # Persistent top-level bar
            root_task = pman.add_task('Walk Media Folders', total=len(root_items), transient=False)

            max_roots = int(self.walk_config.get('max_concurrent_root_walks', 3))
            root_sem = asyncio.Semaphore(max_roots)

            async def _walk_one_root(item):
                async with root_sem:
                    await self._walk_node_async([item], pman, stats, max_depth=initial_depth, root_bar=True)
                # advance only when subtree complete (guaranteed by walker return)
                pman.advance(root_task, 1)

            tasks = [asyncio.create_task(_walk_one_root(item)) for item in root_items]
            await asyncio.gather(*tasks, return_exceptions=False)

        return stats

    # -------------------------
    # Core async walk with subtree-completion progress (DAG-safe)
    # -------------------------

    async def _walk_node_async(self, roots, pman, stats, max_depth=None, root_bar=True):
        import asyncio

        graph = self.graph
        perquery_limit = self.walk_config['perquery_limit']
        attempts = self.walk_config['query_attempts']
        max_req = int(self.walk_config.get('max_concurrent_requests', 20))
        max_parents = int(self.walk_config.get('max_concurrent_parents', 10))
        page_prefetch = bool(self.walk_config.get('page_prefetch', True))

        info_update_interval = float(self.walk_config.get('info_update_interval', 0.5))
        large_threshold = int(self.walk_config.get('large_folder_threshold', 500))

        sem = asyncio.Semaphore(max_req)

        # Behavior / pruning (same intent as before)
        type_add_blocklist = {'UserView', 'CollectionFolder'}
        type_recurse_blocklist = {'Audio', 'Episode'}

        fields = self._coerce_item_fields(self.fields)

        class Frame(typing.NamedTuple):
            item: dict
            depth: int
            is_root: bool
            parent_id: typing.Optional[str]

        q: asyncio.Queue[Frame] = asyncio.Queue()

        roots_list = roots if isinstance(roots, (list, tuple)) else [roots]
        for r in roots_list:
            await q.put(Frame(r, 0, True, None))

        # In-flight de-dup: prevents concurrent expand of same node
        if self._inflight_lock is None:
            self._inflight_lock = asyncio.Lock()
        inflight = self._inflight
        inflight_lock = self._inflight_lock

        # Progress / completion propagation
        # meta[folder_id] = {
        #   'pending_primary_folders': int,  # folders whose completion we are waiting for (only primary children)
        #   'progress_task': task_id|None,  # transient per-folder bar
        #   'parent_id': parent_id|None,    # primary parent for completion propagation
        #   'is_root': bool,
        #   'total_children': int,
        # }
        meta = {}
        meta_lock = asyncio.Lock()

        # child_folder_id -> primary_parent_id
        primary_parent = {}
        primary_lock = asyncio.Lock()

        completed_folders = set()
        completed_lock = asyncio.Lock()

        last_info_update = ub.Timer().tic()

        def should_recurse(item: dict) -> bool:
            if not item.get('IsFolder', False):
                return False
            if item.get('Type') in type_recurse_blocklist:
                return False
            return True

        async def fetch_children(parent_id: str):
            parent_item = graph.nodes[parent_id]['item']
            first = await self._safe_user_items_async(
                parent=parent_item, offset=0, perquery_limit=perquery_limit,
                fields=fields, attempts=attempts, sem=sem,
            )
            items = list(first.get('Items', []))
            total = int(first.get('TotalRecordCount', len(items)))

            if (not page_prefetch) or len(items) >= total:
                offset = len(items)
                while offset < total:
                    page = await self._safe_user_items_async(
                        parent=parent_item, offset=offset, perquery_limit=perquery_limit,
                        fields=fields, attempts=attempts, sem=sem,
                    )
                    page_items = page.get('Items', [])
                    items.extend(page_items)
                    offset += len(page_items)
                return items, total

            tasks = []
            offset = len(items)
            while offset < total:
                tasks.append(asyncio.create_task(self._safe_user_items_async(
                    parent=parent_item, offset=offset, perquery_limit=perquery_limit,
                    fields=fields, attempts=attempts, sem=sem,
                )))
                offset += perquery_limit

            for fut in asyncio.as_completed(tasks):
                page = await fut
                items.extend(page.get('Items', []))
            return items, total

        def add_node_if_missing(child: dict):
            cid = child['Id']
            if cid not in graph.nodes:
                graph.add_node(cid, item=child, properties=dict(expanded=False))

        def add_edge(parent_id: str, child_id: str):
            if not graph.has_edge(parent_id, child_id):
                graph.add_edge(parent_id, child_id)

        async def ensure_folder_task(node_id: str, total_children: int, is_root_folder: bool):
            if pman is None:
                return None
            create = is_root_folder or (total_children >= large_threshold)
            if not create:
                return None
            desc = graph.nodes[node_id]['item'].get('Name', '<no-name>')
            return pman.add_task(f'Walk {desc}', total=total_children, transient=True)

        async def advance_folder_task(folder_id: str, n: int):
            if pman is None:
                return
            async with meta_lock:
                task_id = meta.get(folder_id, {}).get('progress_task', None)
            if task_id is not None and n:
                pman.advance(task_id, n)

        async def maybe_finish_folder(folder_id: str):
            """
            If folder_id has no pending primary child folders, mark it complete and propagate completion
            upward (advance parent by 1 if parent is waiting on this as a primary child).
            """
            async with meta_lock:
                m = meta.get(folder_id, None)
                if m is None:
                    return
                pending = m['pending_primary_folders']
                parent_id = m['parent_id']
                task_id = m['progress_task']

            if pending != 0:
                return

            async with completed_lock:
                if folder_id in completed_folders:
                    return
                completed_folders.add(folder_id)

            # Ensure its own bar is at 100% (it should be, but guard against logic slips)
            if pman is not None and task_id is not None:
                # Best-effort: set completed=total if needed
                try:
                    # rich doesn't expose "set to total" cleanly, but update(completed=total) works.
                    async with meta_lock:
                        total = meta.get(folder_id, {}).get('total_children', None)
                    if total is not None:
                        pman.update(task_id, completed=total)
                except Exception:
                    pass

                # mark task as gone so nobody tries to update it later
                async with meta_lock:
                    if folder_id in meta:
                        meta[folder_id]['progress_task'] = None

                # then remove it from rich
                if pman is not None and task_id is not None:
                    pman.remove_task(task_id)

            # Propagate completion to primary parent (if any)
            if parent_id is not None:
                await advance_folder_task(parent_id, 1)
                async with meta_lock:
                    if parent_id in meta:
                        meta[parent_id]['pending_primary_folders'] -= 1
                await maybe_finish_folder(parent_id)

        async def expand_folder(frame: Frame):
            parent_item = frame.item
            parent_id = parent_item['Id']

            # If we won't expand due to max_depth, treat as completed unit for its parent if needed.
            if max_depth is not None and frame.depth >= max_depth:
                if frame.parent_id is not None:
                    # This frame represents a child folder that was counted as primary pending by its parent.
                    await advance_folder_task(frame.parent_id, 1)
                    async with meta_lock:
                        if frame.parent_id in meta:
                            meta[frame.parent_id]['pending_primary_folders'] -= 1
                    await maybe_finish_folder(frame.parent_id)
                return

            # Mark expanded in graph
            graph.nodes[parent_id]['properties']['expanded'] = True

            stats['latest_name'] = parent_item.get('Name', None)
            stats['latest_path'] = parent_item.get('Path', None)

            # Special features for Series/Season (same as prior)
            if parent_item.get('Type') in {'Series', 'Season'}:
                special_features = await self._special_features_async(parent_id, sem=sem)
                if special_features:
                    special_features_id = parent_id + '/SpecialFeatures'
                    special_parent = {'Name': 'Special Features', 'Id': special_features_id, 'Type': 'SpecialFeatures'}
                    if special_parent['Id'] not in graph:
                        graph.add_node(special_parent['Id'], item=special_parent, properties=dict(expanded=True))
                        stats['node_types'][special_parent['Type']] += 1
                    if not graph.has_edge(parent_id, special_parent['Id']):
                        graph.add_edge(parent_id, special_parent['Id'])
                        stats['edge_types'][(parent_item.get('Type'), special_parent['Type'])] += 1
                    for special in special_features:
                        if special['Id'] not in graph:
                            graph.add_node(special['Id'], item=special, properties=dict(expanded=False))
                            stats['node_types'][special.get('Type')] += 1
                        if not graph.has_edge(special_parent['Id'], special['Id']):
                            graph.add_edge(special_parent['Id'], special['Id'])
                            stats['edge_types'][('SpecialFeatures', special.get('Type'))] += 1

            # Fetch direct children
            children, total_children = await fetch_children(parent_id)

            # Ensure meta entry and transient bar creation
            is_root_folder = frame.is_root
            task_id = await ensure_folder_task(parent_id, total_children, is_root_folder=is_root_folder)

            async with meta_lock:
                meta[parent_id] = {
                    'pending_primary_folders': 0,
                    'progress_task': task_id,
                    'parent_id': frame.parent_id,
                    'is_root': is_root_folder,
                    'total_children': total_children,
                }

            # Process direct children in a way that guarantees EXACTLY total_children progress units.
            immediate_units = 0
            primary_pending = 0

            for child in children:
                cid = child['Id']
                ctype = child.get('Type')
                ptype = parent_item.get('Type')

                # Always count this direct child as "processed" for progress totals
                # (either immediate or later via primary subtree completion).
                if ctype in type_add_blocklist:
                    # Skipped from graph, but counts immediately for progress.
                    immediate_units += 1
                    continue

                # Graph: add nodes/edges, and update stats
                if cid in graph.nodes:
                    stats['nondag_edge_types'][(ptype, ctype)] += 1
                    add_edge(parent_id, cid)
                else:
                    add_node_if_missing(child)
                    add_edge(parent_id, cid)
                    stats['node_types'][ctype] += 1
                    stats['edge_types'][(ptype, ctype)] += 1

                # Decide recursion and progress attribution
                if should_recurse(child):
                    # If this child folder is already fully completed, parent gets immediate unit.
                    async with completed_lock:
                        already_completed = cid in completed_folders
                    if already_completed:
                        immediate_units += 1
                        continue

                    # Primary-parent ownership: only the primary parent waits for subtree completion.
                    async with primary_lock:
                        existing_parent = primary_parent.get(cid, None)
                        if existing_parent is None:
                            primary_parent[cid] = parent_id
                            owned_by_me = True
                        else:
                            owned_by_me = (existing_parent == parent_id)

                    if owned_by_me:
                        primary_pending += 1
                        await q.put(Frame(child, frame.depth + 1, False, parent_id))
                    else:
                        # Not my primary child: count immediately so my bar can still reach 100%.
                        immediate_units += 1
                else:
                    immediate_units += 1

            # Apply progress + pending counts
            if immediate_units:
                await advance_folder_task(parent_id, immediate_units)

            async with meta_lock:
                if parent_id in meta:
                    meta[parent_id]['pending_primary_folders'] = primary_pending

            # Stats total = discovered direct children (same spirit as before)
            stats['total'] += len(children)

            # This folder might finish immediately if no pending primary folders
            await maybe_finish_folder(parent_id)

            # Periodic info update
            if pman is not None and last_info_update.toc() > info_update_interval:
                async with meta_lock:
                    pending_folders = sum(v.get('pending_primary_folders', 0) for v in meta.values())
                    active_folders = sum(1 for v in meta.values() if v.get('pending_primary_folders', 0) > 0)
                info = {
                    'latest_name': stats.get('latest_name'),
                    'latest_path': stats.get('latest_path'),
                    'total_items_seen': stats.get('total', 0),
                    'pending_primary_folders_sum': pending_folders,
                    'active_folders': active_folders,
                    'queue': q.qsize(),
                    'node_types': dict(stats['node_types']),
                    'edge_types': dict(stats['edge_types']),
                    'nondag_edge_types': dict(stats['nondag_edge_types']),
                }
                pman.update_info(ub.urepr(info, nl=2))
                last_info_update.tic()

        async def worker(worker_id: int):
            while True:
                frame = await q.get()
                pid = frame.item['Id']
                try:
                    # If already expanded, then if it is also completed, a waiting parent may need credit.
                    try:
                        if graph.nodes[pid]['properties'].get('expanded', False):
                            # If already completed, its parent (if waiting as primary) should get credit.
                            async with completed_lock:
                                done = pid in completed_folders
                            if done and frame.parent_id is not None:
                                await advance_folder_task(frame.parent_id, 1)
                                async with meta_lock:
                                    if frame.parent_id in meta:
                                        meta[frame.parent_id]['pending_primary_folders'] -= 1
                                await maybe_finish_folder(frame.parent_id)
                            continue
                    except KeyError:
                        continue

                    # In-flight de-dup: only one worker expands a given node at a time.
                    async with inflight_lock:
                        if pid in inflight:
                            # Another worker is doing it; do NOT decrement parent's pending here,
                            # because this frame represents a primary wait, and completion will
                            # propagate when the expander finishes (via meta[parent_id]).
                            continue
                        if graph.nodes[pid]['properties'].get('expanded', False):
                            continue
                        inflight.add(pid)

                    try:
                        await expand_folder(frame)
                    finally:
                        async with inflight_lock:
                            inflight.discard(pid)
                finally:
                    q.task_done()

        workers = [asyncio.create_task(worker(i)) for i in range(max_parents)]

        try:
            await q.join()
        finally:
            for w in workers:
                w.cancel()
            results = await asyncio.gather(*workers, return_exceptions=True)
            for r in results:
                if isinstance(r, asyncio.CancelledError):
                    continue
                if isinstance(r, BaseException):
                    raise r

        # Final guard: force completion checks for roots
        for r in roots_list:
            await maybe_finish_folder(r['Id'])

        # Final guard: if any remaining transient tasks exist, force them to 100 and remove.
        # This should be a no-op in correct operation, but it prevents UI "hanging" even if
        # a rare edge case slipped through.
        if pman is not None:
            async with meta_lock:
                leftovers = list(meta.items())
            for fid, m in leftovers:
                task_id = m.get('progress_task', None)
                if task_id is None:
                    continue
                total = m.get('total_children', None)
                if total is not None:
                    pman.update(task_id, completed=total)
                pman.remove_task(task_id)

    # -------------------------
    # Network calls
    # -------------------------

    async def _safe_user_items_async(self, parent, offset, perquery_limit, fields, attempts=1, verbose=False, sem=None):
        import asyncio
        import traceback

        client = self.client
        parent_id = parent['Id']
        parent_name = parent.get('Name', '<no-name>')
        parent_path = parent.get('Path', None)

        last_err = None
        for attempt in range(1, attempts + 1):
            try:
                kwargs = {
                    'parent_id': parent_id,
                    'recursive': False,
                    'fields': fields,
                    'limit': perquery_limit,
                    'start_index': offset,
                }
                user_id = getattr(client, 'user_id', None)
                if user_id is not None:
                    kwargs['user_id'] = user_id

                if sem is None:
                    resp = await client.api.items.get_items.asyncio_detailed(**kwargs)
                else:
                    async with sem:
                        resp = await client.api.items.get_items.asyncio_detailed(**kwargs)

                assert resp.status_code == 200
                return resp.parsed.to_dict()

            except Exception as err:
                last_err = err
                print(
                    f'[MediaGraph] get_items failed (attempt {attempt}/{attempts}) '
                    f'parent={parent_name!r} id={parent_id} path={parent_path!r} '
                    f'offset={offset} limit={perquery_limit} err={type(err).__name__}: {err}'
                )
                if verbose:
                    traceback.print_exc()
                await asyncio.sleep(0.5 * (2 ** (attempt - 1)))

        raise Exception(
            f'[MediaGraph] giving up on parent={parent_name!r} id={parent_id} after {attempts} attempts'
        ) from last_err

    async def _special_features_async(self, item_id, sem=None):
        client = self.client
        if sem is None:
            resp = await client.api.user_library.get_special_features.asyncio_detailed(item_id=item_id)
        else:
            async with sem:
                resp = await client.api.user_library.get_special_features.asyncio_detailed(item_id=item_id)
        assert resp.status_code == 200
        return [f.to_dict() for f in resp.parsed]

    def _coerce_item_fields(self, fields):
        from jellyfin_apiclient_python.openapi._generated.models.item_fields import ItemFields
        if fields is None:
            return None
        if isinstance(fields, str):
            if ',' in fields:
                fields = [f.strip() for f in fields.split(',') if f.strip()]
            else:
                fields = [fields]

        coerced = []
        for field in fields:
            if isinstance(field, ItemFields):
                coerced.append(field)
                continue
            field_name = str(field)
            try:
                coerced.append(ItemFields(field_name))
            except Exception:
                if self._DEBUG:
                    print(f'[MediaGraph] Unknown ItemField: {field_name!r}, skipping')
                continue
        return coerced

    # -------------------------
    # Labels (UNCHANGED)
    # -------------------------

    def _update_graph_labels(self, sources=None):
        """
        Update the rich text representation of select items in the graph.
        """

        glyphs = {
            'FILE_FOLDER': '📁',
            'OPEN_FILE_FOLDER': '📂',
            'FOLD': '🗀',
            'OPEN_FOLDER': '🗁',
            'BEAMED_SIXTEENTH_NOTES': '♬',
            'MOVIE_CAMERA': '🎥',
            'TELEVISION': '📺',
            'FILM_FRAMES': '🎞',
        }

        url = self.client.base_url
        graph = self.graph

        reachable_nodes = reachable(graph, sources)

        # Relabel Graph
        for node in reachable_nodes:
            node_data = graph.nodes[node]
            item = node_data['item']
            properties = node_data['properties']
            expanded = properties.get('expanded', False)
            glyph_key = 'OPEN_FILE_FOLDER' if expanded else 'FILE_FOLDER'
            type_glyph = glyphs[glyph_key]

            if item['Type'] == 'Folder':
                color = 'blue'
            elif item['Type'] == 'CollectionFolder':
                color = 'blue'
            elif item['Type'] == 'Series':
                color = 'cyan'
            elif item['Type'] == 'Season':
                color = 'yellow'
            elif item['Type'] == 'MusicAlbum':
                color = 'cyan'
            elif item['Type'] == 'MusicArtist':
                color = 'cyan'
            elif item['Type'] == 'Episode':
                color = None
                type_glyph = glyphs['TELEVISION']
            elif item['Type'] == 'Video':
                color = None
                type_glyph = glyphs['FILM_FRAMES']
            elif item['Type'] == 'Movie':
                color = None
                type_glyph = glyphs['MOVIE_CAMERA']
            elif item['Type'] == 'Audio':
                color = None
                type_glyph = glyphs['BEAMED_SIXTEENTH_NOTES']
            else:
                color = None

            if color is not None:
                color_part1 = f'[{color}]'
                color_part2 = f'[/{color}]'
            else:
                color_part1 = ''
                color_part2 = ''

            namerep = item['Name']
            path = item.get('Path', None)
            if self.display_config['show_path']:
                if path is not None:
                    namerep = item['Name'] + ' - ' + path

            item_id_link = f'{url}/web/index.html#!/details?id={item["Id"]}'
            item_id_rep = item["Id"]
            item_id_rep = f'[link={item_id_link}]{item_id_rep}[/link]'

            label = f'{color_part1} {item_id_rep} : {type_glyph} {item["Type"]} - {namerep} {color_part2}'
            node_data['label'] = label

    # -------------------------
    # Printing / searching
    # -------------------------

    def print(self):
        self.print_graph()

    def print_graph(self, sources=None, max_depth=None):
        nx.write_network_text(self.graph, path=rich.print, end='', sources=sources, max_depth=max_depth)

    def print_item(self, node):
        node_data = self.graph.nodes[node]
        item = node_data.get('item', None)
        properties = node_data.get('properties', None)
        rprint(f'node={node}')
        rprint(f'properties = {ub.urepr(properties, nl=1)}')
        rprint(f'item = {ub.urepr(item, nl=1)}')

    def find(self, pattern, data=False, root=None):
        import networkx as nx
        if root is None:
            root = self._cwd
        graph = self.graph
        if root is None:
            nodes = graph.nodes
        else:
            nodes = nx.descendants(graph, root)
        for node in nodes:
            node_data = graph.nodes[node]
            item = node_data['item']
            name = item['Name']
            if pattern in name:
                if data:
                    yield node, node_data
                else:
                    yield node

    def find_one(self, pattern, data=False, root=None):
        matches = list(self.find(pattern, data=data, root=root))
        if not matches:
            raise KeyError(f'find_one({pattern!r}) found no matches')
        if len(matches) > 1:
            raise KeyError(f'find_one({pattern!r}) found {len(matches)} matches, expected exactly one')
        return matches[0]


def reachable(graph, sources=None):
    if sources is None:
        yield from graph.nodes
    else:
        import networkx as nx
        seen = set()
        for source in sources:
            if source in seen:
                continue
            for node in nx.dfs_preorder_nodes(graph, source):
                seen.add(node)
                yield node


def rprint(*args):
    try:
        import rich
        rich.print(*args)
    except ImportError:
        print(*args)


class _RichWalkProgress:
    """Rich progress manager with optional transient tasks and an info panel."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self._active = False
        self.info_panel = None
        self.progress = None
        self.live = None
        self.group = None
        self._Panel = None
        self._setup()

    def _setup(self):
        from rich.console import Group
        from rich.live import Live
        from rich.panel import Panel
        from rich.progress import Progress as RichProgress
        from rich.progress import BarColumn, TextColumn, SpinnerColumn, ProgressColumn, Text
        import rich.progress as rich_progress

        class ProgressRateColumn(ProgressColumn):
            def render(self, task) -> Text:
                itps = task.finished_speed or task.speed
                if itps is not None:
                    rate_format = '4.2f' if itps > .001 else 'g'
                    text = ('{:' + rate_format + '} Hz').format(itps)
                else:
                    text = '?'
                return Text(text, style='progress.data.speed')

        self._Panel = Panel

        self.progress = RichProgress(
            TextColumn("{task.description}"),
            SpinnerColumn(),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            rich_progress.MofNCompleteColumn(),
            ProgressRateColumn(),
            'eta',
            rich_progress.TimeRemainingColumn(),
            'total',
            rich_progress.TimeElapsedColumn(),
            transient=False,
        )
        self.group = Group(self.progress)
        self.live = Live(self.group, refresh_per_second=20)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        self.stop(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)

    def start(self):
        if self.enabled and not self._active:
            self._active = True
            self.live.__enter__()

    def stop(self, **kw):
        if self.enabled and self._active:
            if not kw:
                kw = {'exc_type': None, 'exc_val': None, 'exc_tb': None}
            self.live.__exit__(**kw)
            self._active = False

    def add_task(self, desc, total=None, transient=True):
        if not self.enabled:
            return None
        task_id = self.progress.add_task(description=desc, total=total)
        return task_id

    def update(self, task_id, **kw):
        if not (self.enabled and task_id is not None):
            return
        try:
            self.progress.update(task_id, **kw)
        except KeyError:
            # task already removed; ignore
            return

    def advance(self, task_id, n=1):
        if not (self.enabled and task_id is not None):
            return
        try:
            self.progress.update(task_id, advance=n)
        except KeyError:
            return

    def remove_task(self, task_id):
        if not (self.enabled and task_id is not None):
            return
        try:
            self.progress.remove_task(task_id)
        except KeyError:
            return


    def update_info(self, text):
        if not self.enabled:
            return
        if self.info_panel is None:
            self.info_panel = self._Panel(text)
            self.group.renderables.insert(0, self.info_panel)
        else:
            self.info_panel.renderable = text

