import React, { useMemo } from 'react';
import { Table2, Download, Hash, Tag } from 'lucide-react';

export interface ResultItem {
    material_type: string;
    tag: string;
    sheet: string;
    page: string;
    description: string;
    location: string;
    confidence: string;
}

interface ResultsPanelProps {
    results: ResultItem[];
    onDownload: () => void;
}

export default function ResultsPanel({ results, onDownload }: ResultsPanelProps) {
    const { totalTags, tagCounts } = useMemo(() => {
        const total = results.length;
        const counts: Record<string, number> = {};
        for (const r of results) {
            const tag = r.tag || '—';
            counts[tag] = (counts[tag] ?? 0) + 1;
        }
        const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
        return { totalTags: total, tagCounts: sorted };
    }, [results]);

    return (
        <div className="fixed inset-0 w-screen h-screen bg-[#0a0e17] z-[3000] overflow-y-auto p-10 animate-in fade-in zoom-in duration-300">
            <div className="flex items-center justify-between mb-4">
                <div className="text-[1.1rem] font-semibold flex items-center gap-2">
                    <Table2 className="w-5 h-5" /> Extraction Results
                </div>
                <button
                    onClick={onDownload}
                    className="p-[10px_20px] bg-[rgba(0,212,255,0.15)] border border-[#00D4FF] rounded-lg text-[#00D4FF] font-semibold flex items-center gap-2 transition-all hover:bg-[#00D4FF] hover:text-black"
                >
                    <Download className="w-4 h-4" /> Download CSV
                </button>
            </div>

            {/* Summary: total tags + count per tag */}
            <div className="mb-6 flex flex-wrap items-center gap-4">
                <div className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-[#111827] border border-[#30363d]">
                    <Hash className="w-5 h-5 text-[#00D4FF]" />
                    <span className="text-[#8b949e] text-sm">Total tags found</span>
                    <span className="font-mono font-bold text-[#00D4FF] text-lg">{totalTags}</span>
                </div>
                <div className="flex items-center gap-2 flex-wrap">
                    <Tag className="w-5 h-5 text-[#818CF8] flex-shrink-0" />
                    <span className="text-[#8b949e] text-sm mr-1">Per tag:</span>
                    {tagCounts.map(([tag, count]) => (
                        <span
                            key={tag}
                            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#1a2332] border border-[#30363d] text-sm"
                        >
                            <span className="font-mono font-semibold text-[#00D4FF]">{tag}</span>
                            <span className="text-[#8b949e]">×</span>
                            <span className="font-mono font-bold text-white">{count}</span>
                        </span>
                    ))}
                </div>
            </div>

            <div className="w-full overflow-x-auto">
                <table className="w-full min-w-[600px] border-collapse">
                    <thead>
                        <tr>
                            {['Material Type', 'Tag', 'Sheet', 'Page', 'Description', 'Location', 'Confidence'].map((h) => (
                                <th key={h} className="p-[10px_14px] text-left border-b border-[#30363d] bg-[#1a2332] text-xs text-[#8b949e] uppercase">
                                    {h}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {results.map((r, i) => (
                            <tr key={i} className="border-b border-[#30363d] hover:bg-[#111827]">
                                <td className="p-[10px_14px]">{r.material_type}</td>
                                <td className="p-[10px_14px]">
                                    <span className="bg-[rgba(0,212,255,0.15)] text-[#00D4FF] px-2.5 py-1 rounded font-mono font-semibold text-sm">
                                        {r.tag}
                                    </span>
                                </td>
                                <td className="p-[10px_14px]">{r.sheet}</td>
                                <td className="p-[10px_14px]">{r.page}</td>
                                <td className="p-[10px_14px]">{r.description}</td>
                                <td className="p-[10px_14px]">
                                    <span className="font-mono text-sm text-[#818CF8] block max-w-[250px] break-words">
                                        {r.location}
                                    </span>
                                </td>
                                <td className="p-[10px_14px]">{r.confidence}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
