"""
Async Jellyfin media graph crawler with "filesystem-like" navigation.

Key goals:
- Keep the *exact* same NetworkX graph labeling logic as before (see _update_graph_labels).
- Allow "browse" behavior: only crawl top-level media folders initially, then expand folders on demand.
- Make async walking robust (no deadlocks) and more debuggable.

Notable fixes / changes vs the previous async version:
- **Fixed deadlock**: workers no longer `return` when they encounter a duplicate; they `continue`.
- **Fixed browse regression**: previously a shallow initial crawl could mark nodes as "started" and
  prevent later expansion. Now we track in-flight expansions separately and only de-dup on the
  node's `properties['expanded']` flag.
- Improved structure: isolated concurrency / de-dup logic and added optional debug tracing.

The rich label strings and how they're computed are intentionally unchanged.
"""

import typing
import rich
import ubelt as ub
import networkx as nx
import progiter


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
            # "browse mode" default: only add top-level folders on setup
            # - None means crawl everything recursively
            # - 0 means only initialize root nodes (no children)
            # - 1 means root + direct children, etc...
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

            # Only show per-folder progress bars when the folder is big (except roots)
            'min_progress_total': 200,

            # Periodic info panel update interval (seconds)
            'info_update_interval': 1.0,
        }
        self.display_config = {
            'show_path': False,
        }
        self._cwd = None
        self._cwd_children = None
        self._media_root_nodes = None
        self._DEBUG = False

        # In-flight expansion de-dup across concurrent walks / open_node calls.
        # IMPORTANT: This is distinct from "expanded"; it only prevents multiple
        # concurrent expansions of the same node. Nodes can be expanded later.
        self._inflight = set()
        self._inflight_lock = None  # asyncio.Lock, initialized lazily

        # NOTE: It might not be a great idea to collect all fields by default
        # Things like CumulativeRunTimeTicks might require aggregation
        from jellyfin_apiclient_python.openapi._generated.models.item_fields import ItemFields
        self.fields = set(ItemFields) - {
            ItemFields.CUMULATIVERUNTIMETICKS,
            ItemFields.RECURSIVEITEMCOUNT,
            ItemFields.LOCALTRAILERCOUNT,
            ItemFields.PEOPLE,
        }
        self.fields = [ItemFields.PATH, ItemFields.GENRES, ItemFields.PARENTID]

    def _dbg(self, msg: str):
        if self._DEBUG:
            print(f'[MediaGraph] {msg}')

    @classmethod
    def _run_async(self, coro):
        """Run an async coroutine, creating/managing an event loop if needed.

        - If no loop is running in this thread: uses ``asyncio.run``.
        - If a loop is already running (e.g. Jupyter): runs the coroutine in a
          dedicated background thread with its own event loop.
        """
        import asyncio
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        # Running in an existing loop; create a new loop in a background thread.
        import threading

        result_box = {}
        error_box = {}

        def _thread_main():
            try:
                result_box['result'] = asyncio.run(coro)
            except Exception as ex:  # nocov
                error_box['error'] = ex

        t = threading.Thread(target=_thread_main, daemon=True)
        t.start()
        t.join()

        if 'error' in error_box:
            raise error_box['error']
        return result_box.get('result', None)

    @classmethod
    def ensure_demo_server(cls, reset: bool = False):
        """
        Ensure we have a demo Jellyfin server running for interactive use or tests.

        - If reset=True, destroy any previous container and start fresh.
        - Otherwise reuse an existing container for speed.
        """
        from jellyfin_apiclient_python.demo import JellyfinDockerServer
        server = JellyfinDockerServer(reuse_container=not reset)
        server.start()
        cls._demo_server = server
        return server

    @classmethod
    def demo_client(cls):
        """
        Create a client for demos

        Returns:
            jellyfin_apiclient_python.JellyfinClient
        """
        # TODO: Ensure test environment can spin up a dummy jellyfin server.
        from jellyfin_apiclient_python.openapi.client import Jellyfin
        url = 'http://127.0.1.1:34907'
        username = 'jellyfin-user'
        password = 'jellyfin-pass'

        url="http://192.168.222.38:8096"
        username="jellyfin"
        password=""

        client = Jellyfin(
            base_url=url,
            username=username,
            password=password,
        )
        client.login()
        return client

    def tree(self, max_depth=None):
        """
        Print the graph at the current working directory.
        """
        if self._cwd is None:
            self.print_graph(max_depth=max_depth)
        else:
            self.print_graph(sources=[self._cwd], max_depth=max_depth)

    def ls(self):
        """
        List the children of the current working directory (node) in the graph.
        """
        if self._cwd is None:
            return self._media_root_nodes
        else:
            return self._cwd_children

    def cd(self, node):
        """
        Change the cwd to a specific node, and add its children to the graph if
        they have not already been.
        """
        self._cwd = node
        if node is None:
            self._cwd_children = self._media_root_nodes
        else:
            if not self.graph.nodes[node]['item']['IsFolder']:
                raise Exception('can only cd into a folder')
            self.open_node(node, verbose=0)
            self._cwd_children = list(self.graph.succ[node])

    def __truediv__(self, node):
        self.open_node(node, verbose=1)
        return self

    def setup(self):
        """Populate the initial media folder graph.

        This is a synchronous convenience wrapper that will create/manage its
        own asyncio event loop when needed.
        """
        self._run_async(self.setup_async())
        self._update_graph_labels()

    def open_node(self, node, verbose=0, max_depth=1):
        """Synchronously expand a node in the graph."""
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
        }
        pman = ub.ProgIter(desc='Open Node', verbose=verbose)
        with pman:
            self._run_async(self._walk_node_async([item], pman, stats, max_depth=max_depth))
        self._update_graph_labels(sources=[node_id])

        if verbose:
            self.print_graph([node])
            self.print_item(node)
        return self

    async def setup_async(self):
        """Async version of :meth:`setup`."""
        await self._init_media_folders_async()
        return self

    def _init_media_folders(self):
        """Synchronous wrapper for :meth:`_init_media_folders_async`."""
        return self._run_async(self._init_media_folders_async())

    async def _init_media_folders_async(self):
        """Initialize the graph with the user's top-level media folders and
        optionally pre-walk them.

        This aims to be faithful to the original synchronous implementation,
        but uses the new OpenAPI client and supports async operation.
        """
        client = self.client

        # Initialize graph
        if self._DEBUG:
            print('Initializing, clearing existing DiGraph')
        graph = nx.DiGraph()
        self.graph = graph

        include_collection_types = self.walk_config.get('include_collection_types', None)
        exclude_collection_types = self.walk_config.get('exclude_collection_types', None)
        initial_depth = self.walk_config['initial_depth']

        if self._DEBUG:
            print('Query top level media folder')
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

        # The /Library/MediaFolders endpoint may return either:
        #   (A) actual library items directly in Items
        #   (B) media-folder containers with a Children list of actual items
        for folder in data.get('Items', []):
            candidates = folder.get('Children') or [folder]
            for item in candidates:
                # Normalize type / collection type and apply filters (orig behavior)
                collection_type = item.get('CollectionType', folder.get('CollectionType', None))
                if include_collection_types is not None and collection_type not in include_collection_types:
                    continue
                if exclude_collection_types is not None and collection_type in exclude_collection_types:
                    continue

                # Ensure the node exists
                item_id = item['Id']
                item['type'] = item.get('Type', item.get('type', None))
                if item_id not in graph:
                    graph.add_node(item_id, item=item, properties=dict(expanded=False))

                root_items.append(item)
                root_node_ids.append(item_id)

        self._media_root_nodes = root_node_ids

        # Pre-walk each media root (optional)
        if self._DEBUG:
            print('... top level scan complete, starting media folder walk.')
        pman = _RichWalkProgress(enabled=True)
        with pman:
            # Root-level progress bar (always shown, even for small roots)
            root_task = pman.add_task('Walk Media Folders', total=len(root_items))

            import asyncio
            max_roots = int(self.walk_config.get('max_concurrent_root_walks', 3))
            root_sem = asyncio.Semaphore(max_roots)

            async def _walk_one_root(item):
                async with root_sem:
                    await self._walk_node_async([item], pman, stats, max_depth=initial_depth)
                # Advance after the root walk completes (successfully)
                pman.advance(root_task, 1)

            tasks = [asyncio.create_task(_walk_one_root(item)) for item in root_items]
            # Bubble exceptions like the synchronous version: any failure aborts setup.
            await asyncio.gather(*tasks, return_exceptions=False)

            pman.remove_task(root_task)

        return stats

    async def _walk_node_async(self, roots, pman, stats, max_depth=None):
        """Concurrent async walker that expands nodes using OpenAPI ``asyncio_detailed``.

        Concurrency strategy:
            * Process multiple parent folders concurrently (worker pool).
            * For each parent, fetch remaining pagination pages concurrently.
            * Apply graph mutations in the worker (single-threaded asyncio),
              which is safe because graph ops are atomic in this thread.
        """
        import asyncio
        import typing

        graph = self.graph
        perquery_limit = self.walk_config['perquery_limit']
        attempts = self.walk_config['query_attempts']
        max_req = int(self.walk_config.get('max_concurrent_requests', 20))
        max_parents = int(self.walk_config.get('max_concurrent_parents', 10))
        page_prefetch = bool(self.walk_config.get('page_prefetch', True))

        info_update_interval = float(self.walk_config.get('info_update_interval', 1.0))
        min_progress_total = int(self.walk_config.get('min_progress_total', 200))

        sem = asyncio.Semaphore(max_req)

        # Adding UserViews creates duplicates, and collections can be huge / cyclic
        type_add_blocklist = {'UserView', 'CollectionFolder'}
        # Avoid recursing into media that doesn't have meaningful children
        type_recurse_blocklist = {'Audio', 'Episode'}

        fields = self._coerce_item_fields(self.fields)

        class StackFrame(typing.NamedTuple):
            item: dict
            depth: int
            is_root: bool

        q: asyncio.Queue[StackFrame] = asyncio.Queue()
        for _root in (roots if isinstance(roots, (list, tuple)) else [roots]):
            await q.put(StackFrame(_root, 0, True))

        # In-flight de-dup lock (async context)
        if self._inflight_lock is None:
            self._inflight_lock = asyncio.Lock()
        inflight = self._inflight
        inflight_lock = self._inflight_lock

        # Keep the info panel responsive without flooding refresh
        last_info_update = ub.Timer().tic()

        async def fetch_all_children(parent, *, is_root=False):
            """Fetch all children of a parent.

            Progress behavior:
                * Always show a progress bar for root media folders.
                * For non-root folders, only show a bar if TotalRecordCount is large
                  (>= ``min_progress_total``).
            """
            parent_name = parent.get('Name', '<no-name>')
            folder_task = None

            first = await self._safe_user_items_async(
                parent=parent, offset=0, perquery_limit=perquery_limit,
                fields=fields, attempts=attempts, sem=sem,
            )
            items = list(first.get('Items', []))
            total = first.get('TotalRecordCount', len(items))

            if pman is not None and (is_root or total >= min_progress_total):
                folder_task = pman.add_task(f'Walk {parent_name}', total=total)
                if items:
                    pman.advance(folder_task, len(items))

            if not page_prefetch:
                offset = len(items)
                while offset < total:
                    page = await self._safe_user_items_async(
                        parent=parent, offset=offset, perquery_limit=perquery_limit,
                        fields=fields, attempts=attempts, sem=sem,
                    )
                    page_items = page.get('Items', [])
                    items.extend(page_items)
                    offset += len(page_items)
                    if pman is not None and folder_task is not None and page_items:
                        pman.advance(folder_task, len(page_items))
                return items, total, folder_task

            # Fetch remaining pages concurrently and advance as they complete
            tasks = []
            offset = len(items)
            while offset < total:
                tasks.append(asyncio.create_task(
                    self._safe_user_items_async(
                        parent=parent, offset=offset, perquery_limit=perquery_limit,
                        fields=fields, attempts=attempts, sem=sem,
                    )
                ))
                offset += perquery_limit

            if tasks:
                for fut in asyncio.as_completed(tasks):
                    page = await fut
                    page_items = page.get('Items', [])
                    items.extend(page_items)
                    if pman is not None and folder_task is not None and page_items:
                        pman.advance(folder_task, len(page_items))

            return items, total, folder_task

        async def expand_parent(frame: StackFrame):
            parent = frame.item
            parent_id = parent['Id']

            # Respect max_depth without poisoning future expansions
            if max_depth is not None and frame.depth >= max_depth:
                return

            node_data = graph.nodes[parent_id]
            node_data['properties']['expanded'] = True

            stats['latest_name'] = parent.get('Name', None)
            stats['latest_path'] = parent.get('Path', None)

            folder_task = None

            # Special features for Series/Season
            if parent.get('Type') in {'Series', 'Season'}:
                special_features = await self._special_features_async(parent_id, sem=sem)
                if special_features:
                    special_features_id = parent_id + '/SpecialFeatures'
                    special_parent = {
                        'Name': 'Special Features',
                        'Id': special_features_id,
                        'Type': 'SpecialFeatures',
                    }
                    if special_parent['Id'] not in graph:
                        graph.add_node(special_parent['Id'], item=special_parent, properties=dict(expanded=True))
                    if not graph.has_edge(parent_id, special_parent['Id']):
                        graph.add_edge(parent_id, special_parent['Id'])
                    stats['edge_types'][(parent.get('Type'), special_parent['Type'])] += 1
                    for special in special_features:
                        stats['edge_types'][('SpecialFeatures', special.get('Type'))] += 1
                        if special['Id'] not in graph:
                            graph.add_node(special['Id'], item=special, properties=dict(expanded=False))
                        if not graph.has_edge(special_parent['Id'], special['Id']):
                            graph.add_edge(special_parent['Id'], special['Id'])

            children_items, _total, folder_task = await fetch_all_children(parent, is_root=frame.is_root)

            for child in children_items:
                cid = child['Id']

                # Node already exists (possibly reached via another path).
                # Still add the edge.
                if cid in graph.nodes:
                    stats['nondag_edge_types'][(parent.get('Type'), child.get('Type'))] += 1
                    if not graph.has_edge(parent_id, cid):
                        graph.add_edge(parent_id, cid)
                    if child.get('IsFolder') and child.get('Type') not in type_recurse_blocklist:
                        await q.put(StackFrame(child, frame.depth + 1, False))
                    continue

                if child.get('Type') in type_add_blocklist:
                    continue

                stats['edge_types'][(parent.get('Type'), child.get('Type'))] += 1
                stats['node_types'][child.get('Type')] += 1

                graph.add_node(cid, item=child, properties=dict(expanded=False))
                graph.add_edge(parent_id, cid)

                # Queue child folders for expansion
                if child.get('IsFolder') and child.get('Type') not in type_recurse_blocklist:
                    await q.put(StackFrame(child, frame.depth + 1, False))

            stats['total'] += len(children_items)

            if pman is not None and folder_task is not None:
                pman.remove_task(folder_task)

            # Periodic info update
            if pman is not None:
                if last_info_update.toc() > info_update_interval:
                    pman.update_info(ub.urepr(stats))
                    last_info_update.tic()

        async def worker(worker_id: int):
            while True:
                frame = await q.get()
                pid = frame.item['Id']
                try:
                    # Fast skip if already expanded
                    try:
                        if graph.nodes[pid]['properties'].get('expanded', False):
                            continue
                    except KeyError:
                        # Node should exist, but if not, don't crash workers
                        continue

                    # Respect max_depth early (and don't "reserve" inflight)
                    if max_depth is not None and frame.depth >= max_depth:
                        continue

                    # In-flight de-dup: only one worker should expand a node at a time.
                    async with inflight_lock:
                        if pid in inflight:
                            continue
                        # It might have been expanded while waiting for the lock
                        if graph.nodes[pid]['properties'].get('expanded', False):
                            continue
                        inflight.add(pid)

                    try:
                        await expand_parent(frame)
                    finally:
                        async with inflight_lock:
                            inflight.discard(pid)

                finally:
                    q.task_done()

        if self._DEBUG:
            print('... start async workers')
        workers = [asyncio.create_task(worker(i)) for i in range(max_parents)]
        if self._DEBUG:
            print(f'workers={workers}')

        try:
            await q.join()
        finally:
            # Always cancel workers once the queue is done (or if something errors)
            for w in workers:
                w.cancel()
            # Await workers so any exception inside them is re-raised here
            results = await asyncio.gather(*workers, return_exceptions=True)
            for r in results:
                if isinstance(r, asyncio.CancelledError):
                    continue
                if isinstance(r, BaseException):
                    raise r

        if self._DEBUG:
            print(f'stats={stats}')

    async def _safe_user_items_async(self, parent, offset, perquery_limit, fields, attempts=1, verbose=False, sem=None):
        """Async version of :meth:`_safe_user_items` using OpenAPI ``asyncio_detailed``."""
        import asyncio
        import traceback

        client = self.client
        parent_id = parent['Id']
        parent_name = parent.get('Name', '<no-name>')
        parent_path = parent.get('Path', None)
        total_record_count = parent.get('TotalRecordCount', None)

        if self._DEBUG:
            print(f'Issue query {parent_id=} {offset=} {total_record_count=}: {parent_name=}')

        last_err = None
        for attempt in range(1, attempts + 1):
            try:
                if sem is None:
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
                    resp = await client.api.items.get_items.asyncio_detailed(**kwargs)
                else:
                    async with sem:
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
                        resp = await client.api.items.get_items.asyncio_detailed(**kwargs)
                assert resp.status_code == 200
                return resp.parsed.to_dict()
            except Exception as err:  # nocov
                last_err = err
                print(
                    f'[MediaGraph] user_items_async failed (attempt {attempt}/{attempts}) '
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
        """Fetch special features for a Series/Season."""
        client = self.client
        if sem is None:
            resp = await client.api.user_library.get_special_features.asyncio_detailed(item_id=item_id)
        else:
            async with sem:
                resp = await client.api.user_library.get_special_features.asyncio_detailed(item_id=item_id)
        assert resp.status_code == 200
        return [f.to_dict() for f in resp.parsed]

    def _coerce_item_fields(self, fields):
        """
        Normalize field requests into a list of ItemFields / strings.
        """
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
        # http.config.data['auth.server']

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
                    # namerep = path

            item_id_link = f'{url}/web/index.html#!/details?id={item["Id"]}'
            item_id_rep = item["Id"]
            item_id_rep = f'[link={item_id_link}]{item_id_rep}[/link]'

            label = f'{color_part1} {item_id_rep} : {type_glyph} {item["Type"]} - {namerep} {color_part2}'
            node_data['label'] = label

    def print(self):
        """
        Alias for :func:`MediaGraph.print_graph`.
        """
        self.print_graph()

    def print_graph(self, sources=None, max_depth=None):
        """
        Prints the current state of the media graph to stdout at a particular
        starting point with a specified depth.
        """
        nx.write_network_text(self.graph, path=rich.print, end='', sources=sources, max_depth=max_depth)

    def print_item(self, node):
        node_data = self.graph.nodes[node]
        item = node_data.get('item', None)
        properties = node_data.get('properties', None)
        rprint(f'node={node}')
        rprint(f'properties = {ub.urepr(properties, nl=1)}')
        rprint(f'item = {ub.urepr(item, nl=1)}')

    def find(self, pattern, data=False, root=None):
        """
        Search for a pattern within the current directory.

        Args:
            pattern (str): text to find in the media name.
            data (bool): if True, also return the data dict
            root (str | None): if specified search from this location,
                if unspecified the cwd is used.

        Yields:
            str | Tuple[str, dict]:
                the id of the found item, or the id and its data if
                data=True
        """
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
            # TODO: allow multiple types of patterns (i.e. similar to
            # kwutil.Pattern) to abstract regex, glob, and raw string matching.
            if pattern in name:
                if data:
                    yield node, node_data
                else:
                    yield node

    def find_one(self, pattern, data=False, root=None):
        """
        Find exactly one item matching a pattern.

        Args:
            pattern (str): text to find in the media name.
            data (bool): if True, also return the data dict.
            root (str | None): if specified search from this location,
                if unspecified the cwd is used.

        Returns:
            str | Tuple[str, dict]:
                the unique matching item.

        Raises:
            KeyError:
                if no items match or if multiple items match.
        """
        matches = list(self.find(pattern, data=data, root=root))

        if not matches:
            raise KeyError(f'find_one({pattern!r}) found no matches')

        if len(matches) > 1:
            raise KeyError(
                f'find_one({pattern!r}) found {len(matches)} matches, expected exactly one'
            )

        return matches[0]


def reachable(graph, sources=None):
    if sources is None:
        yield from graph.nodes
    else:
        seen = set()
        for source in sources:
            if source in seen:
                continue
            for node in nx.dfs_preorder_nodes(graph, source):
                seen.add(node)
                yield node


def _find_sources(graph):
    """
    Determine a minimal set of nodes such that the entire graph is reachable
    """
    import networkx as nx
    # For each connected part of the graph, choose at least
    # one node as a starting point, preferably without a parent
    if graph.is_directed():
        # Choose one node from each SCC with minimum in_degree
        sccs = list(nx.strongly_connected_components(graph))
        # condensing the SCCs forms a dag, the nodes in this graph with
        # 0 in-degree correspond to the SCCs from which the minimum set
        # of nodes from which all other nodes can be reached.
        scc_graph = nx.condensation(graph, sccs)
        supernode_to_nodes = {sn: [] for sn in scc_graph.nodes()}
        # Note: the order of mapping differs between pypy and cpython
        # so we have to loop over graph nodes for consistency
        mapping = scc_graph.graph["mapping"]
        for n in graph.nodes:
            sn = mapping[n]
            supernode_to_nodes[sn].append(n)
        sources = []
        for sn in scc_graph.nodes():
            if scc_graph.in_degree[sn] == 0:
                scc = supernode_to_nodes[sn]
                node = min(scc, key=lambda n: graph.in_degree[n])
                sources.append(node)
    else:
        # For undirected graph, the entire graph will be reachable as
        # long as we consider one node from every connected component
        sources = [
            min(cc, key=lambda n: graph.degree[n])
            for cc in nx.connected_components(graph)
        ]
        sources = sorted(sources, key=lambda n: graph.degree[n])
    return sources


def rprint(*args):
    try:
        import rich
        rich.print(*args)
    except ImportError:
        print(*args)


class _RichWalkProgress:
    """Standalone rich progress + info panel for async MediaGraph walking.

    Designed to resemble progiter.manager rich backend style, but without
    progiter, so async code can directly manage per-node tasks.
    """
    def __init__(self, enabled=True):
        self.enabled = enabled
        self._active = False
        self.info_panel = None
        self.progress = None
        self.live = None
        self.group = None
        self._setup()

    def _setup(self):
        from rich.console import Group
        from rich.live import Live
        from rich.panel import Panel
        from rich.progress import Progress as RichProgress
        from rich.progress import BarColumn, TextColumn, SpinnerColumn, ProgressColumn, Text
        import rich.progress as rich_progress

        class ProgressRateColumn(ProgressColumn):
            """Shows iterations / second."""
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
        )
        self.info_panel = None
        self.group = Group(self.progress)
        self.live = Live(self.group)

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

    def add_task(self, desc, total=None):
        if not self.enabled:
            return None
        return self.progress.add_task(description=desc, total=total)

    def update(self, task_id, **kw):
        if self.enabled and task_id is not None:
            self.progress.update(task_id, **kw)

    def advance(self, task_id, n=1):
        if self.enabled and task_id is not None:
            self.progress.update(task_id, advance=n)

    def remove_task(self, task_id):
        if self.enabled and task_id is not None:
            self.progress.remove_task(task_id)

    def update_info(self, text):
        if not self.enabled:
            return
        if self.info_panel is None:
            self.info_panel = self._Panel(text)
            # Insert above progress bars
            self.group.renderables.insert(0, self.info_panel)
        else:
            self.info_panel.renderable = text
