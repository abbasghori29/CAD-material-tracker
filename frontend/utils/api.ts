
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const getApiUrl = (): string => {
    if (typeof window === 'undefined') return '';
    return API_URL;
};

export const getWsUrl = (): string => {
    if (typeof window === 'undefined') return '';

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

    // If API URL is relative, construct absolute WS URL from current host
    if (API_URL.startsWith('/')) {
        return `${protocol}//${window.location.host}/ws`;
    }

    // If API URL is absolute (http/https), replace protocol with ws/wss
    return API_URL.replace(/^https?/, protocol.replace(':', '')) + '/ws';
};
