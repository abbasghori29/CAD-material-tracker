
import React, { useState, useRef } from 'react';
import { FileSpreadsheet, FileCheck, ArrowRight, FileUp, Play, ArrowLeft } from 'lucide-react';
import { clsx } from 'clsx';
import Modal from './ui/Modal';
import { getApiUrl } from '../utils/api';

interface UploadPanelProps {
    onStartProcessing: (path: string, startPage: number, endPage: number) => void;
    onTagsUploaded: () => void;
}

export default function UploadPanel({ onStartProcessing, onTagsUploaded }: UploadPanelProps) {
    const [slide, setSlide] = useState(1);
    const [tagsFile, setTagsFile] = useState<File | null>(null);
    const [pdfFile, setPdfFile] = useState<File | null>(null);
    const [uploadedPdfPath, setUploadedPdfPath] = useState<string>("");
    const [tagsCount, setTagsCount] = useState(0);
    const [pdfPages, setPdfPages] = useState(0);

    // Form inputs
    const [startPage, setStartPage] = useState<string>("");
    const [endPage, setEndPage] = useState<string>("");

    // Modal State
    const [modalOpen, setModalOpen] = useState(false);
    const [modalTitle, setModalTitle] = useState("");
    const [modalFile, setModalFile] = useState("");
    const [uploadProgress, setUploadProgress] = useState(0);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);

    // Refs for hidden inputs
    const tagsInputRef = useRef<HTMLInputElement>(null);
    const pdfInputRef = useRef<HTMLInputElement>(null);

    // --- Handlers ---

    const handleTagsSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            await uploadTags(e.target.files[0]);
        }
    };

    const handleTagsDrop = async (e: React.DragEvent) => {
        e.preventDefault();
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            await uploadTags(e.dataTransfer.files[0]);
        }
    };

    // API URL from centralized config

    const authHeaders = (): Record<string, string> => {
        const token = localStorage.getItem('ab_builders_token');
        return token ? { Authorization: `Bearer ${token}` } : {};
    };

    const uploadWithProgress = (
        url: string,
        formData: FormData,
        onProgress: (percent: number) => void
    ): Promise<{ ok: boolean; data: Record<string, unknown> }> => {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.open('POST', url);
            const headers = authHeaders();
            Object.entries(headers).forEach(([k, v]) => xhr.setRequestHeader(k, v));

            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) {
                    onProgress(Math.round((e.loaded / e.total) * 100));
                }
            };

            xhr.onload = () => {
                try {
                    const data = (xhr.responseText ? JSON.parse(xhr.responseText) : {}) as Record<string, unknown>;
                    resolve({ ok: xhr.status >= 200 && xhr.status < 300, data });
                } catch {
                    reject(new Error('Invalid response'));
                }
            };
            xhr.onerror = () => reject(new Error('Network error'));
            xhr.send(formData);
        });
    };

    const uploadTags = async (file: File) => {
        setErrorMessage(null);
        setModalTitle("Uploading Tag List");
        setModalFile(file.name);
        setUploadProgress(0);
        setModalOpen(true);

        const formData = new FormData();
        formData.append('file', file);
        const apiUrl = getApiUrl();

        try {
            const { ok, data } = await uploadWithProgress(
                `${apiUrl}/upload-tags`,
                formData,
                setUploadProgress
            );
            setUploadProgress(100);

            if (ok && data.success) {
                setTagsFile(file);
                setTagsCount((data.count as number) ?? 0);
                onTagsUploaded();
                setTimeout(() => {
                    setModalOpen(false);
                    setSlide(2);
                }, 500);
            } else {
                setErrorMessage((data.error as string) || 'Tag list upload failed');
                setModalOpen(false);
            }
        } catch (err) {
            console.error(err);
            setErrorMessage('Tag list upload failed. Please try again.');
            setModalOpen(false);
        }
    };

    const handlePdfSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            await uploadPdf(e.target.files[0]);
        }
    };

    const handlePdfDrop = async (e: React.DragEvent) => {
        e.preventDefault();
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            await uploadPdf(e.dataTransfer.files[0]);
        }
    };

    const uploadPdf = async (file: File) => {
        setErrorMessage(null);
        setModalTitle("Uploading PDF");
        setModalFile(file.name);
        setUploadProgress(0);
        setModalOpen(true);

        const formData = new FormData();
        formData.append('file', file);
        const apiUrl = getApiUrl();

        try {
            const { ok, data } = await uploadWithProgress(
                `${apiUrl}/upload`,
                formData,
                setUploadProgress
            );
            setUploadProgress(100);

            if (ok && data.path) {
                setPdfFile(file);
                setPdfPages((data.pages as number) ?? 0);
                setUploadedPdfPath(data.path as string);

                setTimeout(() => {
                    setModalOpen(false);
                }, 500);
            } else {
                setErrorMessage((data.error as string) || 'PDF upload failed');
                setModalOpen(false);
            }
        } catch (err) {
            console.error(err);
            setErrorMessage('PDF upload failed. Please try again.');
            setModalOpen(false);
        }
    };

    return (
        <>
            <Modal isOpen={modalOpen} title={modalTitle} filename={modalFile} progress={uploadProgress} />

            <div className="w-full max-w-[100vw] flex flex-col items-center justify-center p-10 min-h-[calc(100vh-80px)]">
                <div className="w-full max-w-[600px] overflow-hidden">

                    {/* Progress Steps */}
                    <div className="flex justify-center gap-3 mb-[30px]">
                        <div className={clsx("w-10 h-1 rounded-sm transition-all duration-300", slide === 1 ? "bg-[#00D4FF] shadow-[0_0_10px_rgba(0,212,255,0.4)]" : "bg-[#10B981]")}></div>
                        <div className={clsx("w-10 h-1 rounded-sm transition-all duration-300", slide === 2 ? "bg-[#00D4FF] shadow-[0_0_10px_rgba(0,212,255,0.4)]" : "bg-[#30363d]")}></div>
                    </div>

                    <div className="flex w-[200%] transition-transform duration-500 ease-[cubic-bezier(0.4,0,0.2,1)]"
                        style={{ transform: slide === 1 ? 'translateX(0%)' : 'translateX(-50%)' }}>

                        {/* Slide 1 */}
                        <div className="w-1/2 px-5 flex-shrink-0">
                            <div className="bg-[#111827] border border-[#30363d] rounded-[20px] p-10 shadow-[0_20px_60px_rgba(0,0,0,0.4)]">
                                <div className="text-center mb-[30px]">
                                    <h2 className="text-[1.5rem] mb-2 bg-gradient-to-br from-[#00D4FF] to-[#818CF8] bg-clip-text text-transparent font-bold">Step 1: Upload Tag List</h2>
                                    <p className="text-[#8b949e] text-sm">Upload CSV or Excel file (XLSX/XLS) with tags to detect</p>
                                    {!tagsFile && (
                                        <p className="text-[#8b949e] text-xs mt-1">
                                            Upload your tag list to get started.
                                        </p>
                                    )}
                                </div>

                                <div
                                    onClick={() => tagsInputRef.current?.click()}
                                    onDragOver={(e) => e.preventDefault()}
                                    onDrop={handleTagsDrop}
                                    className="border-2 border-dashed border-[#30363d] rounded-2xl p-[50px_30px] text-center cursor-pointer transition-all duration-300 mb-6 bg-[#1a2332] hover:border-[#00D4FF] hover:bg-[rgba(0,212,255,0.15)] hover:shadow-[0_0_30px_rgba(0,212,255,0.4)]"
                                >
                                    <input type="file" ref={tagsInputRef} className="hidden" accept=".csv,.xlsx,.xls" onChange={handleTagsSelect} />
                                    <div className="mb-4 flex justify-center"><FileSpreadsheet className="w-16 h-16 text-[#00D4FF]" /></div>
                                    <div className="text-[1.1rem] font-semibold mb-2">Drop CSV/Excel Here</div>
                                    <div className="text-[#8b949e] text-sm">or click to browse • Supports: CSV, XLSX, XLS</div>
                                </div>

                                {errorMessage && (
                                    <div className="mb-4 rounded-lg border border-[#EF4444]/40 bg-[#EF4444]/10 text-[#FCA5A5] text-sm px-3 py-2 text-left">
                                        {errorMessage}
                                    </div>
                                )}

                                {tagsFile && (
                                    <div className="bg-[#1a2332] p-[14px_18px] rounded-xl flex items-center gap-3 mb-5 border border-[#00D4FF]">
                                        <FileCheck className="w-6 h-6 text-[#10B981]" />
                                        <div className="flex-1">
                                            <div className="font-mono text-sm text-[#00D4FF]">{tagsFile.name}</div>
                                            <div className="text-[#8b949e] text-xs mt-0.5">{tagsCount} tags loaded</div>
                                        </div>
                                    </div>
                                )}

                                <div className="flex justify-end gap-4 mt-[30px]">
                                    <button
                                        disabled={!tagsFile}
                                        onClick={() => setSlide(2)}
                                        className="p-[12px_24px] bg-gradient-to-br from-[#00D4FF] to-[#818CF8] text-black font-semibold rounded-[10px] flex items-center gap-2 transition-all disabled:opacity-40 disabled:cursor-not-allowed hover:shadow-[0_10px_30px_rgba(0,212,255,0.4)]"
                                    >
                                        Next Step <ArrowRight className="w-[18px] h-[18px]" />
                                    </button>
                                </div>
                            </div>
                        </div>

                        {/* Slide 2 */}
                        <div className="w-1/2 px-5 flex-shrink-0">
                            <div className="bg-[#111827] border border-[#30363d] rounded-[20px] p-10 shadow-[0_20px_60px_rgba(0,0,0,0.4)]">
                                <div className="text-center mb-[30px]">
                                    <h2 className="text-[1.5rem] mb-2 bg-gradient-to-br from-[#00D4FF] to-[#818CF8] bg-clip-text text-transparent font-bold">Step 2: Upload CAD Drawings</h2>
                                    <p className="text-[#8b949e] text-sm">Extract material tags from PDF drawings using AI</p>
                                    {!pdfFile && (
                                        <p className="text-[#8b949e] text-xs mt-1">
                                            Upload a PDF to begin processing.
                                        </p>
                                    )}
                                </div>

                                <div
                                    onClick={() => pdfInputRef.current?.click()}
                                    onDragOver={(e) => e.preventDefault()}
                                    onDrop={handlePdfDrop}
                                    className="border-2 border-dashed border-[#30363d] rounded-2xl p-[50px_30px] text-center cursor-pointer transition-all duration-300 mb-6 bg-[#1a2332] hover:border-[#00D4FF] hover:bg-[rgba(0,212,255,0.15)] hover:shadow-[0_0_30px_rgba(0,212,255,0.4)]"
                                >
                                    <input type="file" ref={pdfInputRef} className="hidden" accept=".pdf" onChange={handlePdfSelect} />
                                    <div className="mb-4 flex justify-center"><FileUp className="w-16 h-16 text-[#00D4FF]" /></div>
                                    <div className="text-[1.1rem] font-semibold mb-2">Drop PDF Here</div>
                                    <div className="text-[#8b949e] text-sm">or click to browse</div>
                                </div>

                                {pdfFile && (
                                    <div className="bg-[#1a2332] p-[14px_18px] rounded-xl flex items-center gap-3 mb-5 border border-[#00D4FF]">
                                        <FileCheck className="w-6 h-6 text-[#10B981]" />
                                        <div className="flex-1">
                                            <div className="font-mono text-sm text-[#00D4FF]">{pdfFile.name}</div>
                                            <div className="text-[#8b949e] text-xs mt-0.5">{pdfPages} pages</div>
                                        </div>
                                    </div>
                                )}

                                <div className="flex gap-3 mb-5">
                                    <div className="flex-1">
                                        <label className="block text-xs text-[#8b949e] mb-1.5">Start Page <span className="font-normal opacity-70">(optional)</span></label>
                                        <input
                                            type="number"
                                            min="1"
                                            placeholder="1"
                                            value={startPage}
                                            onChange={(e) => setStartPage(e.target.value)}
                                            className="w-full p-3 bg-[#1a2332] border border-[#30363d] rounded-[10px] text-white font-mono text-sm focus:outline-none focus:border-[#00D4FF] focus:shadow-[0_0_0_3px_rgba(0,212,255,0.15)]"
                                        />
                                    </div>
                                    <div className="flex-1">
                                        <label className="block text-xs text-[#8b949e] mb-1.5">End Page <span className="font-normal opacity-70">(optional)</span></label>
                                        <input
                                            type="number"
                                            min="1"
                                            placeholder={pdfPages ? pdfPages.toString() : "All"}
                                            value={endPage}
                                            onChange={(e) => setEndPage(e.target.value)}
                                            className="w-full p-3 bg-[#1a2332] border border-[#30363d] rounded-[10px] text-white font-mono text-sm focus:outline-none focus:border-[#00D4FF] focus:shadow-[0_0_0_3px_rgba(0,212,255,0.15)]"
                                        />
                                    </div>
                                </div>

                                <div className="flex justify-between gap-4 mt-[30px]">
                                    <button
                                        onClick={() => setSlide(1)}
                                        className="p-[12px_24px] bg-[#1a2332] border border-[#30363d] text-[#f0f6fc] font-semibold rounded-[10px] flex items-center gap-2 transition-all hover:border-[#00D4FF] hover:bg-[rgba(0,212,255,0.15)]"
                                    >
                                        <ArrowLeft className="w-[18px] h-[18px]" /> Previous
                                    </button>
                                    <button
                                        disabled={!pdfFile}
                                        onClick={() => {
                                            if (uploadedPdfPath) {
                                                const s = parseInt(startPage) || 1;
                                                const e = parseInt(endPage) || pdfPages;
                                                onStartProcessing(uploadedPdfPath, s, e);
                                            }
                                        }}
                                        className="p-[12px_24px] bg-gradient-to-br from-[#00D4FF] to-[#818CF8] text-black font-semibold rounded-[10px] flex items-center gap-2 transition-all disabled:opacity-40 disabled:cursor-not-allowed hover:shadow-[0_10px_30px_rgba(0,212,255,0.4)]"
                                    >
                                        <Play className="w-[18px] h-[18px]" /> Start Processing
                                    </button>
                                </div>
                            </div>
                        </div>

                    </div>
                </div>
            </div>
        </>
    );
}