'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { FileSpreadsheet, FileUp, ListOrdered, Download, ArrowRight, BookOpen } from 'lucide-react';

export default function DocsPage() {
    const router = useRouter();
    const [authReady, setAuthReady] = useState(false);

    useEffect(() => {
        const token = typeof window !== 'undefined' ? localStorage.getItem('ab_builders_token') : null;
        if (!token) {
            router.replace('/login');
            return;
        }
        setAuthReady(true);
    }, [router]);

    if (!authReady) {
        return (
            <main className="min-h-screen bg-[#0a0e17] flex items-center justify-center">
                <div className="w-8 h-8 border-2 border-[#00D4FF] border-t-transparent rounded-full animate-spin" />
            </main>
        );
    }

    return (
        <main className="min-h-screen bg-[#0a0e17] text-[#f0f6fc]">
            {/* Background */}
            <div className="fixed inset-0 bg-grid pointer-events-none z-0" aria-hidden />
            <div className="fixed -top-40 -right-40 w-96 h-96 rounded-full bg-[#00D4FF]/5 blur-[120px] pointer-events-none z-0" aria-hidden />
            <div className="fixed -bottom-40 -left-40 w-96 h-96 rounded-full bg-[#818CF8]/5 blur-[120px] pointer-events-none z-0" aria-hidden />

            <div className="relative z-10 max-w-4xl mx-auto px-6 py-12 pb-24">
                {/* Back */}
                <Link
                    href="/"
                    className="inline-flex items-center gap-2 text-[#8b949e] hover:text-[#00D4FF] text-sm font-medium mb-10 transition-colors"
                >
                    <ArrowRight className="w-4 h-4 rotate-180" /> Back to Dashboard
                </Link>

                {/* Hero */}
                <header className="mb-16">
                    <div className="flex items-center gap-4 mb-4">
                        <div className="w-14 h-14 rounded-2xl flex items-center justify-center bg-gradient-to-br from-[#00D4FF] to-[#818CF8] shadow-[0_0_30px_rgba(0,212,255,0.3)]">
                            <BookOpen className="w-7 h-7 text-black" />
                        </div>
                        <div>
                            <h1 className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-[#00D4FF] to-[#818CF8] bg-clip-text text-transparent">
                                Documentation
                            </h1>
                            <p className="text-[#8b949e] mt-1">How to use the AI-Powered Estimation Suite</p>
                        </div>
                    </div>
                </header>

                {/* How to use */}
                <section className="mb-16">
                    <h2 className="text-xl font-semibold text-white flex items-center gap-2 mb-6">
                        <ListOrdered className="w-5 h-5 text-[#00D4FF]" />
                        How to use the system
                    </h2>
                    <div className="space-y-6">
                        <div className="bg-[#111827] border border-[#30363d] rounded-2xl p-6 hover:border-[#00D4FF]/30 transition-colors">
                            <div className="flex items-start gap-4">
                                <div className="w-10 h-10 rounded-xl bg-[#00D4FF]/15 flex items-center justify-center flex-shrink-0">
                                    <FileSpreadsheet className="w-5 h-5 text-[#00D4FF]" />
                                </div>
                                <div>
                                    <h3 className="font-semibold text-white mb-1">Step 1: Upload your tag list</h3>
                                    <p className="text-[#8b949e] text-sm leading-relaxed">
                                        Upload a CSV or Excel file (XLSX/XLS) that lists the material tags you want the system to find in your drawings. The file should have a <strong className="text-[#f0f6fc]">Tags</strong> column (and optionally <strong className="text-[#f0f6fc]">Material Type</strong>). See the data format section below and use the sample sheet to get started.
                                    </p>
                                </div>
                            </div>
                        </div>
                        <div className="bg-[#111827] border border-[#30363d] rounded-2xl p-6 hover:border-[#00D4FF]/30 transition-colors">
                            <div className="flex items-start gap-4">
                                <div className="w-10 h-10 rounded-xl bg-[#818CF8]/15 flex items-center justify-center flex-shrink-0">
                                    <FileUp className="w-5 h-5 text-[#818CF8]" />
                                </div>
                                <div>
                                    <h3 className="font-semibold text-white mb-1">Step 2: Upload your CAD PDF</h3>
                                    <p className="text-[#8b949e] text-sm leading-relaxed">
                                        Upload the PDF containing your CAD drawings. You can optionally set a start and end page range. Then click <strong className="text-[#f0f6fc]">Start Processing</strong>. The system will detect drawings, extract text, match your tags, and stream progress in real time. When done, download the results as CSV.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Data the system expects */}
                <section className="mb-16">
                    <h2 className="text-xl font-semibold text-white flex items-center gap-2 mb-6">
                        <FileSpreadsheet className="w-5 h-5 text-[#00D4FF]" />
                        What data the system expects
                    </h2>
                    <div className="bg-[#111827] border border-[#30363d] rounded-2xl p-6 space-y-6">
                        <div>
                            <h3 className="font-medium text-white mb-2">Tag list (CSV or Excel)</h3>
                            <ul className="text-[#8b949e] text-sm space-y-2 list-disc list-inside">
                                <li><strong className="text-[#f0f6fc]">Formats:</strong> CSV, XLSX, or XLS</li>
                                <li><strong className="text-[#f0f6fc]">Required:</strong> A column named <strong className="text-[#00D4FF]">Tags</strong> (or the first column is used as tags)</li>
                                <li><strong className="text-[#f0f6fc]">Optional:</strong> A second column for <strong className="text-[#00D4FF]">Material Type</strong> (e.g. Steel, Concrete)</li>
                                <li>Tag values are normalized to uppercase; empty or invalid rows are skipped</li>
                            </ul>
                        </div>
                        <div>
                            <h3 className="font-medium text-white mb-2">CAD drawings (PDF)</h3>
                            <p className="text-[#8b949e] text-sm">
                                A single PDF file containing one or more pages of CAD drawings. The system extracts full-page images, detects drawing regions, runs OCR, and matches text against your tag list. Results include sheet name, page, location, and confidence.
                            </p>
                        </div>
                    </div>
                </section>

                {/* Download sample */}
                <section className="mb-8">
                    <h2 className="text-xl font-semibold text-white flex items-center gap-2 mb-6">
                        <Download className="w-5 h-5 text-[#00D4FF]" />
                        Sample input sheet
                    </h2>
                    <div className="bg-gradient-to-br from-[#111827] to-[#1a2332] border border-[#30363d] rounded-2xl p-8 text-center relative overflow-hidden">
                        <div className="absolute inset-0 bg-gradient-to-r from-[#00D4FF]/5 to-[#818CF8]/5 pointer-events-none" />
                        <p className="text-[#8b949e] mb-6 relative">
                            Use this Excel template to see the expected format. Fill in your tags and optional material types, then upload it in Step 1.
                        </p>
                        <a
                            href="/sample_input_sheet.xlsx"
                            download="sample_input_sheet.xlsx"
                            className="relative inline-flex items-center gap-2 px-6 py-3.5 rounded-xl bg-gradient-to-r from-[#00D4FF] to-[#818CF8] text-black font-semibold shadow-[0_0_25px_rgba(0,212,255,0.35)] hover:shadow-[0_0_35px_rgba(0,212,255,0.45)] transition-all hover:scale-[1.02]"
                        >
                            <Download className="w-5 h-5" />
                            Download sample Excel sheet
                        </a>
                    </div>
                </section>

                {/* Footer note */}
                <p className="text-center text-[#8b949e] text-sm">
                    Need help? Contact your Absolute Builders administrator.
                </p>
            </div>
        </main>
    );
}
