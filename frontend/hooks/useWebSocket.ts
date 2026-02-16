
import { useEffect, useRef, useState, useCallback } from 'react';
import { getWsUrl, getApiUrl } from '../utils/api';

export type JobStatus = 'idle' | 'running' | 'complete' | 'error';

interface UseWebSocketOptions {
    onMessage: (data: any) => void;
    onOpen?: () => void;
    onClose?: () => void;
    onError?: (error: Event) => void;
}

export function useWebSocket({ onMessage, onOpen, onClose, onError }: UseWebSocketOptions) {
    const ws = useRef<WebSocket | null>(null);
    const [projectId, setProjectId] = useState<string | null>(null);
    const [status, setStatus] = useState<JobStatus>('idle');
    const [isConnected, setIsConnected] = useState(false);

    const reconnectAttempts = useRef(0);
    const maxReconnectAttempts = 999;
    const messageQueue = useRef<any[]>([]);

    // Use refs for callbacks to avoid stale closures and unstable dependencies
    const onMessageRef = useRef(onMessage);
    const onOpenRef = useRef(onOpen);
    const onCloseRef = useRef(onClose);
    const onErrorRef = useRef(onError);
    const statusRef = useRef(status);

    // Track if component is truly mounted (survives Strict Mode)
    const mountedRef = useRef(false);

    // Keepalive & Visibility Refs
    const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);
    const wakeLockRef = useRef<any>(null);

    // Keep refs in sync with latest props/state
    useEffect(() => { onMessageRef.current = onMessage; }, [onMessage]);
    useEffect(() => { onOpenRef.current = onOpen; }, [onOpen]);
    useEffect(() => { onCloseRef.current = onClose; }, [onClose]);
    useEffect(() => { onErrorRef.current = onError; }, [onError]);
    useEffect(() => { statusRef.current = status; }, [status]);



    // 1. Wake Lock
    const requestWakeLock = async () => {
        if ('wakeLock' in navigator) {
            try {
                wakeLockRef.current = await (navigator as any).wakeLock.request('screen');
                console.log('[WAKELOCK] Screen wake lock acquired');
                wakeLockRef.current.addEventListener('release', () => {
                    console.log('[WAKELOCK] Screen wake lock released');
                });
            } catch (err) {
                console.log('[WAKELOCK] Wake lock request failed:', err);
            }
        }
    };

    // 3. Keepalive Logic (Dynamic Interval)
    const setupKeepalive = useCallback(() => {
        if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);

        const isVisible = !document.hidden;
        const intervalMs = isVisible ? 5000 : 2000;

        pingIntervalRef.current = setInterval(() => {
            if (ws.current?.readyState === WebSocket.OPEN) {
                try {
                    ws.current.send(JSON.stringify({ action: 'ping' }));
                } catch (e) {
                    console.error('[KEEPALIVE] Ping failed', e);
                }
            }
        }, intervalMs);
    }, []);

    // Connect function - stable reference thanks to refs
    const connect = useCallback(() => {
        // Don't connect if already connected or connecting
        if (ws.current && (ws.current.readyState === WebSocket.OPEN || ws.current.readyState === WebSocket.CONNECTING)) {
            return;
        }

        let wsUrl = getWsUrl();
        if (!wsUrl) {
            console.error('[WS] No WebSocket URL configured');
            return;
        }
        const token = localStorage.getItem('ab_builders_token');
        if (token) {
            const separator = wsUrl.includes('?') ? '&' : '?';
            wsUrl = `${wsUrl}${separator}token=${encodeURIComponent(token)}`;
        }
        console.log('[WS] Connecting to:', wsUrl);

        try {
            ws.current = new WebSocket(wsUrl);
        } catch (e) {
            console.error('[WS] Creation failed', e);
            return;
        }

        ws.current.onopen = () => {
            console.log('[WS] Connected');
            setIsConnected(true);
            reconnectAttempts.current = 0;
            if (onOpenRef.current) onOpenRef.current();

            // Flush queue
            while (messageQueue.current.length > 0) {
                const msg = messageQueue.current.shift();
                ws.current?.send(JSON.stringify(msg));
                console.log('[WS] Sent queued message:', msg.action);
            }

            // Setup keepalive
            setupKeepalive();

            // Request Wake Lock on connection
            requestWakeLock();
        };

        ws.current.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                // Handle job logic state internally
                if (data.type === 'job_created' || data.type === 'reconnect_summary') {
                    const jid = data.job_id;
                    if (jid) {
                        setProjectId(jid);
                        setStatus('running');
                        localStorage.setItem('cad_tracker_job_id', jid);
                        requestWakeLock();
                    }
                } else if (data.type === 'complete') {
                    setStatus('complete');
                    localStorage.removeItem('cad_tracker_job_id');
                    if (wakeLockRef.current) wakeLockRef.current.release().catch(() => { });
                    // Stop keepalive pings and close connection so server can release resources
                    if (pingIntervalRef.current) {
                        clearInterval(pingIntervalRef.current);
                        pingIntervalRef.current = null;
                    }
                    const socket = ws.current;
                    if (socket?.readyState === WebSocket.OPEN) {
                        setTimeout(() => {
                            socket.close(1000, 'Job complete');
                        }, 2000);
                    }
                } else if (data.type === 'error') {
                    // Check if fatal?
                }

                onMessageRef.current(data);
            } catch (e) {
                console.error('[WS] Parse error:', e);
            }
        };

        ws.current.onclose = (event) => {
            console.log('[WS] Closed:', event.code, event.reason);
            setIsConnected(false);
            if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
            if (onCloseRef.current) onCloseRef.current();

            // Auto-reconnect if we have a running job
            const savedJob = localStorage.getItem('cad_tracker_job_id');
            const currentStatus = statusRef.current;
            const shouldReconnect = (savedJob && currentStatus !== 'complete');

            if (shouldReconnect && reconnectAttempts.current < maxReconnectAttempts) {
                const timeout = Math.min(1000 * (reconnectAttempts.current + 1), 5000);
                reconnectAttempts.current++;
                console.log(`[WS] Reconnecting in ${timeout}ms... (Attempt ${reconnectAttempts.current})`);
                setTimeout(connect, timeout);
            }
        };

        ws.current.onerror = (error) => {
            console.error('[WS] Error:', error);
            if (onErrorRef.current) onErrorRef.current(error);
        };

    }, [setupKeepalive]); // Stable deps only - callbacks via refs

    const sendMessage = useCallback((msg: any) => {
        if (ws.current?.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify(msg));
        } else {
            console.log('[WS] Queuing message:', msg.action);
            messageQueue.current.push(msg);
            if (!ws.current || ws.current.readyState === WebSocket.CLOSED) {
                connect();
            }
        }
    }, [connect]);

    // 2. Visibility Handling
    useEffect(() => {
        const handleVisibilityChange = () => {
            const isVisible = !document.hidden;

            if (isVisible) {
                console.log('[VISIBILITY] Tab became visible');
                if (statusRef.current === 'running') requestWakeLock();

                if (ws.current && ws.current.readyState === WebSocket.OPEN) {
                    try { ws.current.send(JSON.stringify({ action: 'ping' })); } catch (e) { }
                } else {
                    const savedJob = localStorage.getItem('cad_tracker_job_id');
                    if (savedJob && statusRef.current !== 'complete') {
                        console.log('[VISIBILITY] Reconnecting after tab became visible');
                        connect();
                    }
                }
            } else {
                console.log('[VISIBILITY] Tab hidden - keeping connection alive');
            }

            setupKeepalive();
        };

        document.addEventListener('visibilitychange', handleVisibilityChange);
        return () => {
            document.removeEventListener('visibilitychange', handleVisibilityChange);
            if (wakeLockRef.current) wakeLockRef.current.release().catch(() => { });
        };
    }, [connect, setupKeepalive]);

    // Mount/unmount lifecycle + auto-resume
    useEffect(() => {
        mountedRef.current = true;

        const checkAndResume = async () => {
            const savedJobId = localStorage.getItem('cad_tracker_job_id');
            if (!savedJobId) return;

            const apiUrl = getApiUrl();

            try {
                const res = await fetch(`${apiUrl}/api/jobs/${savedJobId}`);
                const data = await res.json();

                // If component unmounted during the fetch, bail out
                if (!mountedRef.current) return;

                if (data.error || data.status === 'completed' || data.status === 'failed') {
                    localStorage.removeItem('cad_tracker_job_id');
                    return;
                }

                if (data.status === 'running') {
                    console.log('[AUTO-RESUME] Found active job:', savedJobId);
                    setProjectId(savedJobId);
                    setStatus('running');
                    connect();
                    sendMessage({ action: 'subscribe', job_id: savedJobId });
                }
            } catch (e) {
                console.error('[AUTO-RESUME] Failed to check job:', e);
            }
        };

        checkAndResume();

        return () => {
            mountedRef.current = false;

            // Only close if truly unmounting (not React Strict Mode dev cycle)
            // Use a small delay to distinguish Strict Mode remount from real unmount
            const currentWs = ws.current;
            const currentPingInterval = pingIntervalRef.current;

            setTimeout(() => {
                if (!mountedRef.current && currentWs) {
                    console.log('[WS] Unmounting - closing');
                    currentWs.close(1000, 'Component unmounted');
                }
                if (!mountedRef.current && currentPingInterval) {
                    clearInterval(currentPingInterval);
                }
            }, 100);

            if (wakeLockRef.current) wakeLockRef.current.release().catch(() => { });
        };
    }, []); // Run once on mount

    return {
        connect,
        sendMessage,
        projectId,
        status,
        isConnected
    };
}
