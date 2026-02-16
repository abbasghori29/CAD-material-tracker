
/**
 * In production, API calls go through Next.js rewrites (same origin).
 * In development (port 3000), we also use rewrites, so relative URLs work everywhere.
 *
 * If NEXT_PUBLIC_API_URL is explicitly set, it overrides everything.
 */
export const getApiUrl = (): string => {
    if (typeof window === 'undefined') return '';

    // If explicitly set, use it (e.g. for custom setups)
    if (process.env.NEXT_PUBLIC_API_URL) {
        return process.env.NEXT_PUBLIC_API_URL;
    }

    // Default: use relative URL — Next.js rewrites will proxy to backend
    return '';
};

export const getWsUrl = (): string => {
    if (typeof window === 'undefined') return '';

    const explicitApi = process.env.NEXT_PUBLIC_API_URL;

    if (explicitApi) {
        // If explicit API URL set, derive WS from it
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        if (explicitApi.startsWith('/')) {
            return `${protocol}//${window.location.host}/ws`;
        }
        return explicitApi.replace(/^https?/, protocol.replace(':', '')) + '/ws';
    }

    // Default: same-origin WebSocket via Next.js rewrite
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/ws`;
};
