(args) => {
    const rid = args.rid;
    window.__AISTUDIO__ = window.__AISTUDIO__ || {};
    const streams = window.__AISTUDIO__.streams = window.__AISTUDIO__.streams || {};
    const aborts = window.__AISTUDIO__.streamAbort = window.__AISTUDIO__.streamAbort || {};
    const existing = streams[rid];
    if (existing && existing.abort) {
        try { existing.abort(); } catch (e) {}
    }

    const state = {
        reader: null,
        abortController: null,
        startedAt: Date.now(),
        bytes: 0,
        gotStatus: false,
        attempts: 0,
    };
    streams[rid] = state;

    function push(event) {
        event.rid = rid;
        if (typeof window.__aistudio_stream_push__ === 'function') {
            try {
                window.__aistudio_stream_push__(JSON.stringify(event));
            } catch (e) {}
        }
    }

    const timeoutMs = (args.timeout || 60) * 1000;
    let timeoutId = setTimeout(() => {
        push({type: 'error', message: 'timeout'});
        doAbort();
    }, timeoutMs);

    function resetTimeout() {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
            push({type: 'error', message: 'timeout'});
            doAbort();
        }, timeoutMs);
    }

    function cleanup() {
        timeoutId && clearTimeout(timeoutId);
        timeoutId = null;
        try {
            if (window.__AISTUDIO__?.streams) delete window.__AISTUDIO__.streams[rid];
            if (window.__AISTUDIO__?.streamAbort) delete window.__AISTUDIO__.streamAbort[rid];
        } catch (e) {}
    }
    const doAbort = function() {
        cleanup();
        if (state.reader) {
            try { state.reader.cancel(); } catch (e) {}
        }
        if (state.abortController) {
            try { state.abortController.abort(); } catch (e) {}
        }
    };
    aborts[rid] = doAbort;
    state.abort = doAbort;

    const headers = Object.assign({}, args.headers || {});

    function fetchErrorEvent(err, phase) {
        return {
            type: 'error',
            message: (err && err.message) || 'network error',
            errName: (err && err.name) || 'unknown',
            phase: phase,
            attempts: state.attempts,
            elapsedMs: Date.now() - state.startedAt,
            gotStatus: state.gotStatus,
            bytes: state.bytes,
        };
    }

    function startFetch() {
        state.attempts += 1;
        const abortController = new AbortController();
        state.abortController = abortController;
        window.fetch(args.url, {
            method: 'POST',
            headers: headers,
            body: args.body,
            credentials: 'include',
            signal: abortController.signal,
        }).then(response => {
            resetTimeout();
            state.gotStatus = true;
            push({type: 'status', status: response.status || 0});

            if (!response.body) {
                push({type: 'done'});
                cleanup();
                return;
            }

            const reader = response.body.getReader();
            state.reader = reader;
            const decoder = new TextDecoder('utf-8');

            function readLoop() {
                reader.read().then(({done, value}) => {
                    if (done) {
                        try {
                            const remaining = decoder.decode();
                            if (remaining) {
                                push({type: 'chunk', text: remaining});
                            }
                        } catch (e) {}
                        push({type: 'done'});
                        cleanup();
                        return;
                    }
                    if (value) {
                        resetTimeout();
                        state.bytes += value.byteLength || 0;
                        const text = decoder.decode(value, {stream: true});
                        if (text) {
                            push({type: 'chunk', text: text});
                        }
                    }
                    readLoop();
                }).catch(err => {
                    if (err && err.name === 'AbortError') {
                        push({type: 'aborted'});
                    } else {
                        push(fetchErrorEvent(err, 'read'));
                    }
                    cleanup();
                });
            }
            readLoop();
        }).catch(err => {
            if (err && err.name === 'AbortError') {
                push({type: 'aborted'});
                cleanup();
                return;
            }
            // 页内安全重试：初始 fetch 网络级失败（TypeError，如 Failed to fetch，
            // 典型诱因是连接被重置 / DNS 抖动 / 页面导航竞争）发生在收到响应头之前，
            // 此时尚未产生任何输出字节，重试不会重复计费。
            if (err && err.name === 'TypeError' && !state.gotStatus && state.attempts < 2) {
                push({
                    type: 'probe',
                    probe: 'fetch_net_error_retry',
                    attempt: state.attempts,
                    errName: err.name,
                    errMessage: (err && err.message) || '',
                    elapsedMs: Date.now() - state.startedAt,
                });
                setTimeout(() => {
                    if (state.gotStatus || !streams[rid]) return;
                    resetTimeout();
                    startFetch();
                }, 800);
                return;
            }
            push(fetchErrorEvent(err, 'fetch'));
            cleanup();
        });
    }
    startFetch();
}
