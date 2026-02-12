
'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Header from '../components/Header';
import Footer from '../components/Footer';
import UploadPanel from '../components/UploadPanel';
import ProcessingView, { LogEntry, Drawing } from '../components/ProcessingView';
import ResultsPanel, { ResultItem } from '../components/ResultsPanel';
import { useWebSocket } from '../hooks/useWebSocket';

const getApiUrl = () => {
    if (typeof window === 'undefined') return '';
    return window.location.port === '3000'
        ? `http://${window.location.hostname}:8000`
        : '';
};

export default function Home() {
    const router = useRouter();
    const [view, setView] = useState<'upload' | 'processing' | 'results'>('upload');
    const [tagsUploaded, setTagsUploaded] = useState(false);

    // User state
    const [userName, setUserName] = useState<string | null>(null);
    const [userEmail, setUserEmail] = useState<string | null>(null);
    const [authReady, setAuthReady] = useState(false);

    // Processing State
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [drawings, setDrawings] = useState<Drawing[]>([]);
    const [stats, setStats] = useState({ pages: 0, drawings: 0, tags: 0 });
    const [currentPage, setCurrentPage] = useState(0);
    const [totalPageCount, setTotalPageCount] = useState(0);
    const [currentSheetName, setCurrentSheetName] = useState('N/A');
    const [lastImageUrl, setLastImageUrl] = useState<string | null>(null);
    const [progress, setProgress] = useState(0);
    const [results, setResults] = useState<ResultItem[]>([]);

    // Auth guard + fetch user
    useEffect(() => {
        const token = typeof window !== 'undefined'
            ? localStorage.getItem('ab_builders_token')
            : null;
        if (!token) {
            router.replace('/login');
            return;
        }

        // Fetch current user info
        fetch(`${getApiUrl()}/auth/me`, {
            headers: { Authorization: `Bearer ${token}` },
        })
            .then(res => {
                if (!res.ok) throw new Error('Unauthorized');
                return res.json();
            })
            .then(data => {
                setUserName(data.full_name || null);
                setUserEmail(data.email || null);
                setAuthReady(true);
            })
            .catch(() => {
                // Token expired or invalid
                localStorage.removeItem('ab_builders_token');
                router.replace('/login');
            });
    }, [router]);

    const handleLogout = () => {
        localStorage.removeItem('ab_builders_token');
        localStorage.removeItem('cad_tracker_job_id');
        router.replace('/login');
    };

    // WebSocket Handler
    const handleMessage = useCallback((data: any) => {
        switch (data.type) {
            case 'response':
                if (data.status === 'running') {
                    setView('processing');
                    setStats({
                        pages: data.pages_processed || 0,
                        drawings: 0,
                        tags: data.results_count || 0
                    });
                    setCurrentPage(data.current_page || 0);
                    setTotalPageCount(data.total_pages || 0);
                    addLog('success', `Resumed job ${data.job_id}`);
                }
                break;

            case 'reconnect_summary':
                setView('processing');
                setStats({
                    pages: data.pages_processed,
                    drawings: 0,
                    tags: data.results_count
                });
                setCurrentPage(data.current_page);
                setTotalPageCount(data.total_pages);
                addLog('success', `Resumed job ${data.job_id}`);
                break;

            case 'info':
                setTotalPageCount(data.total_pages);
                break;

            case 'page_start':
                setCurrentPage(data.page);
                if (data.sheet && data.sheet !== 'N/A') {
                    setCurrentSheetName(data.sheet);
                    addLog('info', `Processing page ${data.page} - Sheet ${data.sheet}`);
                } else {
                    addLog('info', `Processing page ${data.page}`);
                }
                if (data.total > 0) {
                    setProgress((data.progress / data.total) * 100);
                }
                setStats(prev => ({ ...prev, pages: data.progress }));
                setDrawings([]);
                break;

            case 'full_page': {
                let imgUrl = data.image_url;
                if (imgUrl && !imgUrl.startsWith('http')) {
                    imgUrl = `${getApiUrl()}${imgUrl}`;
                }
                setLastImageUrl(imgUrl);
                if (data.annotated) {
                    addLog('success', `Found ${data.drawing_count} drawings on page ${data.page}`);
                }
                break;
            }

            case 'drawing': {
                setStats(prev => ({ ...prev, drawings: prev.drawings + 1 }));
                let drawImgUrl = data.image_url;
                if (drawImgUrl && !drawImgUrl.startsWith('http')) {
                    drawImgUrl = `${getApiUrl()}${drawImgUrl}`;
                }
                const newDrawing: Drawing = {
                    index: data.index,
                    confidence: data.confidence,
                    image_url: drawImgUrl,
                    location: 'Extracting...',
                    tags: []
                };
                setDrawings(prev => [newDrawing, ...prev]);
                break;
            }

            case 'ocr_result':
                setDrawings(prev => prev.map(d => {
                    if (d.index === data.drawing_index) {
                        return {
                            ...d,
                            tags: data.tags_found || [],
                            location: data.location || 'Not detected'
                        };
                    }
                    return d;
                }));
                break;

            case 'tag_match':
                // Tag matches are streamed per detection, but the final, de-duplicated
                // count is only known after processing completes. We now set the
                // visible tag count from the 'complete' event instead.
                break;

            case 'log':
                addLog(data.level as any, data.message);
                break;

            case 'complete':
                addLog('success', `Complete! Found ${data.total_tags} material tags`);
                if (typeof data.total_tags === 'number') {
                    setStats(prev => ({ ...prev, tags: data.total_tags }));
                }
                if (data.results) {
                    setResults(data.results);
                    setTimeout(() => {
                        setView('results');
                    }, 1000);
                }
                break;

            case 'error':
                addLog('error', data.message);
                break;
        }
    }, []);

    // Ref for sendMessage to break circular dependency
    const sendMessageRef = useRef<((msg: any) => void) | null>(null);

    const handleOpen = useCallback(() => {
        const savedJobId = localStorage.getItem('cad_tracker_job_id');
        if (savedJobId) {
            sendMessageRef.current?.({ action: 'subscribe', job_id: savedJobId });
        }
    }, []);

    const { connect, sendMessage, projectId, status } = useWebSocket({
        onMessage: handleMessage,
        onOpen: handleOpen,
    });

    useEffect(() => { sendMessageRef.current = sendMessage; }, [sendMessage]);

    const addLog = (level: LogEntry['level'], message: string) => {
        const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
        setLogs(prev => [...prev, { time, level, message }]);
    };

    const handleStartProcessing = (path: string, start: number, end: number) => {
        if (!path) return;
        setView('processing');
        setLogs([]);
        setDrawings([]);
        setStats({ pages: 0, drawings: 0, tags: 0 });
        connect();
        sendMessage({
            action: 'process',
            pdf_path: path,
            start_page: start,
            end_page: end
        });
    };

    const handleDownload = () => {
        const token = localStorage.getItem('ab_builders_token');
        const url = `${getApiUrl()}/download`;
        if (token) {
            fetch(url, {
                headers: { Authorization: `Bearer ${token}` },
            })
                .then(res => res.blob())
                .then(blob => {
                    const href = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = href;
                    a.download = 'material_results.csv';
                    document.body.appendChild(a);
                    a.click();
                    a.remove();
                    window.URL.revokeObjectURL(href);
                })
                .catch(() => {
                    window.location.href = url;
                });
        } else {
            window.location.href = url;
        }
    };

    // Show nothing until auth is confirmed (prevents flash)
    if (!authReady) {
        return (
            <main className="min-h-screen bg-[#0a0e17] flex items-center justify-center">
                <div className="flex flex-col items-center gap-4">
                    <div className="w-10 h-10 border-2 border-[#00D4FF] border-t-transparent rounded-full animate-spin" />
                    <span className="text-sm text-[#8b949e]">Loading...</span>
                </div>
            </main>
        );
    }

    return (
        <main className="min-h-screen bg-[#0a0e17] text-[#f0f6fc] font-sans flex flex-col">
            <div className="bg-grid" />

            <Header
                userName={userName}
                userEmail={userEmail}
                onLogout={handleLogout}
            />

            <div className="flex-1 relative">
                {view === 'upload' && (
                    <UploadPanel
                        onTagsUploaded={() => setTagsUploaded(true)}
                        onStartProcessing={handleStartProcessing}
                    />
                )}

                {view === 'processing' && (
                    <ProcessingView
                        logs={logs}
                        drawings={drawings}
                        stats={stats}
                        currentPage={currentPage}
                        totalPageCount={totalPageCount}
                        currentSheetName={currentSheetName}
                        lastImageUrl={lastImageUrl}
                        progress={progress}
                        status={status}
                    />
                )}

                {view === 'results' && (
                    <ResultsPanel
                        results={results}
                        onDownload={handleDownload}
                    />
                )}
            </div>

            {/* Footer only on upload view; processing/results are full-screen */}
            {view === 'upload' && <Footer />}
        </main>
    );
}
