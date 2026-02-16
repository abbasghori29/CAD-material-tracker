
import React, { useEffect, useRef } from 'react';
import { FileImage, Scan, Terminal, FileScan } from 'lucide-react';
import { clsx } from 'clsx';
import { JobStatus } from '../hooks/useWebSocket';

export interface LogEntry {
    time: string;
    level: 'info' | 'success' | 'warning' | 'error';
    message: string;
}

export interface Drawing {
    index: number;
    confidence: string;
    image_url: string;
    location?: string;
    tags: string[];
}

interface ProcessingViewProps {
    logs: LogEntry[];
    drawings: Drawing[];
    stats: { pages: number; drawings: number; tags: number };
    currentPage: number;
    totalPageCount: number; // total pages in PDF
    currentSheetName: string;
    lastImageUrl: string | null;
    progress: number; // 0-100
    status: JobStatus;
}

export default function ProcessingView({
    logs,
    drawings,
    stats,
    currentPage,
    totalPageCount,
    currentSheetName,
    lastImageUrl,
    progress,
    status
}: ProcessingViewProps) {

    const logsRef = useRef<HTMLDivElement>(null);
    const drawingsRef = useRef<HTMLDivElement>(null);

    let statusLine = '';
    if (status === 'running' && totalPageCount > 0 && currentPage > 0) {
        // Show absolute page number AND progress index (e.g. "Page 76 (1 of 7)")
        statusLine = `Processing page ${currentPage} (${stats.pages} of ${totalPageCount})`;
    } else if (status === 'running') {
        statusLine = 'Analyzing drawings...';
    } else if (status === 'complete') {
        statusLine = 'Processing complete';
    } else if (status === 'error') {
        statusLine = 'Processing failed';
    } else {
        statusLine = 'Waiting to start...';
    }

    // Auto-scroll logs
    useEffect(() => {
        if (logsRef.current) {
            logsRef.current.scrollTop = logsRef.current.scrollHeight;
        }
    }, [logs]);

    // Auto-scroll drawings when new one added
    useEffect(() => {
        if (drawingsRef.current) {
            // scroll to bottom smoothly
            // Check if user is already at bottom or close to it before forcing?
            // For this app, auto-scroll is preferred.
            // drawingsRef.current.scrollTop = drawingsRef.current.scrollHeight;
        }
    }, [drawings.length]);

    // Icons mapping for logs
    const logIcons = {
        info: '→',
        success: '✓',
        warning: '⚠',
        error: '✗'
    };

    const logColors = {
        info: 'text-[#00D4FF]',
        success: 'text-[#10B981]',
        warning: 'text-[#F59E0B]',
        error: 'text-[#EF4444]'
    };

    return (
        <div className="fixed top-[90px] left-5 right-5 bottom-5 bg-[#0a0e17] rounded-[20px] p-5 z-[1000] grid grid-cols-2 gap-5 overflow-visible">

            {/* Left Panel: Full Page Preview */}
            <div className="flex flex-col bg-[#111827] rounded-xl border border-[#30363d] overflow-hidden min-w-0 relative z-10">
                <div className="p-[16px_20px] border-b border-[#30363d] flex items-center justify-between bg-[#1a2332]">
                    <div className="font-semibold flex items-center gap-2">
                        <FileImage className="w-[18px] h-[18px]" /> Full Page Preview
                    </div>
                    <div className="font-mono text-[#00D4FF] text-sm">
                        {currentPage > 0 ? `Page ${currentPage} ${currentSheetName !== 'N/A' ? `(Sheet ${currentSheetName})` : ''}` : 'Waiting...'}
                    </div>
                </div>

                {/* Progress Bar */}
                <div className="h-1 bg-[#0a0e17]">
                    <div className="h-full bg-gradient-to-r from-[#00D4FF] to-[#818CF8] transition-all duration-300" style={{ width: `${progress}%` }}></div>
                </div>

                <div className="flex-1 p-5 flex items-center justify-center bg-[#0a0e17] overflow-auto">
                    {lastImageUrl ? (
                        <img src={lastImageUrl} alt="Page Preview" className="max-w-full max-h-full rounded-lg shadow-lg" />
                    ) : (
                        <div className="text-center text-[#8b949e]">
                            <div className="opacity-50 mb-4"><FileScan className="w-12 h-12 mx-auto" /></div>
                            <div>Waiting for page...</div>
                        </div>
                    )}
                </div>
            </div>

            {/* Right Panel: Detected Drawings & Logs */}
            <div className="flex flex-col gap-5 h-full overflow-hidden min-w-0 relative z-10">

                {/* Header */}
                <div className="flex flex-col flex-1 min-h-0">
                    <div className="flex items-center justify-between p-[16px_20px] bg-[#1a2332] border-t border-x border-[#30363d] rounded-t-xl">
                        <div className="font-semibold flex items-center gap-2">
                            <Scan className="w-[18px] h-[18px]" /> Detected Drawings
                        </div>
                        <div className="font-mono text-[#00D4FF] text-sm">{drawings.length} drawings</div>
                    </div>

                    {/* Stats Bar + Status — Tags only shown after dedup (when > 0); per-drawing tags are in each card */}
                    <div className="flex items-center justify-between gap-5 p-[12px_20px] bg-[#0a0e17] border-b border-x border-[#30363d] flex-shrink-0">
                        <div className="flex gap-5">
                            <div className="flex items-center gap-2">
                                <span className="font-mono text-[1.2rem] font-bold text-[#00D4FF]">{stats.pages}</span>
                                <span className="text-xs text-[#8b949e]">Pages</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="font-mono text-[1.2rem] font-bold text-[#00D4FF]">{stats.drawings}</span>
                                <span className="text-xs text-[#8b949e]">Drawings</span>
                            </div>
                            {stats.tags > 0 && (
                                <div className="flex items-center gap-2">
                                    <span className="font-mono text-[1.2rem] font-bold text-[#00D4FF]">{stats.tags}</span>
                                    <span className="text-xs text-[#8b949e]">Tags</span>
                                </div>
                            )}
                        </div>
                        <div className="text-xs text-[#8b949e] font-mono text-right">
                            {statusLine}
                        </div>
                    </div>

                    {/* Drawings List */}
                    <div ref={drawingsRef} className="flex-1 overflow-y-auto overflow-x-hidden p-5 border-x border-[#30363d] bg-transparent">
                        {drawings.length === 0 ? (
                            <div className="text-center text-[#8b949e] py-10">Waiting for detections...</div>
                        ) : (
                            drawings.map((draw, i) => (
                                <div key={`${draw.index}-${i}`} id={`drawing-${draw.index}`} className="bg-[#1a2332] border border-[#ff3b3b] shadow-[0_0_25px_rgba(255,59,59,0.15)] rounded-2xl mb-4 overflow-hidden transition-all duration-300">
                                    <div className="p-[14px_18px] bg-[#0a0e17] flex items-center justify-between text-base font-semibold">
                                        <span className="font-bold text-[#FF3B3B]">Drawing #{draw.index}</span>
                                        <span className="font-mono text-[#8b949e] text-sm font-medium">{draw.confidence}</span>
                                    </div>
                                    <div className="p-4 bg-[#0a0e17] flex justify-center items-center">
                                        <img src={draw.image_url} alt={`Drawing ${draw.index}`} className="max-w-full max-h-[280px] w-auto rounded-lg border-2 border-[#30363d] shadow-lg object-contain" />
                                    </div>

                                    {/* Location */}
                                    <div className="p-[14px_18px] bg-[#0a0e17] border-t border-[#30363d] text-[0.95rem] flex items-center gap-2">
                                        <span className="text-[#8b949e] font-medium">📍 Location:</span>
                                        {draw.location ? (
                                            <span className="text-[#818CF8] font-mono font-semibold">{draw.location}</span>
                                        ) : (
                                            <span className="text-[#8b949e]">Extracting...</span>
                                        )}
                                    </div>

                                    {/* Tags */}
                                    <div className="p-[16px_18px] flex flex-wrap gap-2.5 bg-[#1a2332]">
                                        {draw.tags.length > 0 ? (
                                            draw.tags.map((tag, idx) => (
                                                <span key={idx} className="bg-[rgba(0,255,157,0.15)] text-[#00D4FF] px-3.5 py-2 rounded-lg text-[0.95rem] font-semibold border-2 border-[rgba(0,255,157,0.3)] shadow-[0_2px_8px_rgba(0,255,157,0.2)]">
                                                    {tag}
                                                </span>
                                            ))
                                        ) : (
                                            <span className="p-[8px_14px] rounded-lg font-mono text-[0.95rem] font-semibold bg-[#0a0e17] text-[#8b949e] border border-[#30363d]">Scanning...</span>
                                        )}
                                    </div>
                                </div>
                            ))
                        ).reverse()}
                    </div>
                </div>

                {/* Logs Panel - Fixed Height */}
                <div className="h-[200px] flex-shrink-0 border-t border-[#30363d] bg-[#1a2332] rounded-2xl flex flex-col overflow-hidden">
                    <div className="p-[10px_16px] border-b border-[#30363d] flex items-center justify-between bg-[#1a2332]">
                        <div className="text-sm font-semibold flex items-center gap-2">
                            <Terminal className="w-4 h-4" /> Live Logs
                        </div>
                    </div>
                    <div ref={logsRef} className="flex-1 overflow-y-auto overflow-x-hidden p-3 font-mono text-[0.85rem] scroll-smooth">
                        {logs.map((log, i) => (
                            <div key={i} className="py-1 flex gap-2.5 border-b border-[#30363d] last:border-0 text-[#f0f6fc]">
                                <span className="text-[#8b949e] min-w-[70px]">{log.time}</span>
                                <span className={clsx("w-4", logColors[log.level])}>{logIcons[log.level]}</span>
                                <span>{log.message}</span>
                            </div>
                        ))}
                    </div>
                </div>

            </div>
        </div>
    );
}
