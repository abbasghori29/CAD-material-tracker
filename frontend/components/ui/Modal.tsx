
import React from 'react';

interface ModalProps {
    isOpen: boolean;
    title: string;
    filename: string;
    progress: number;
}

export default function Modal({ isOpen, title, filename, progress }: ModalProps) {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black/85 backdrop-blur-md z-[2000] flex items-center justify-center p-10 overflow-y-auto">
            <div className="bg-[#111827] border border-[#30363d] rounded-[20px] p-8 text-center min-w-[280px] max-w-[70%] w-auto max-h-[70vh] overflow-y-auto m-auto shadow-[0_20px_60px_rgba(0,0,0,0.4)]">
                {/* Loader Animation */}
                <div className="w-[60px] h-[60px] mx-auto mb-6 relative">
                    <div className="absolute w-full h-full border-4 border-transparent border-t-[#00D4FF] border-r-[#818CF8] rounded-full animate-[spin_1.2s_cubic-bezier(0.5,0,0.5,1)_infinite]"></div>
                    <div className="absolute w-[80%] h-[80%] top-[10%] left-[10%] border-4 border-transparent border-t-[#818CF8] border-r-[#00D4FF] rounded-full animate-[spin_0.8s_reverse_infinite]"></div>
                    <div className="absolute w-[60%] h-[60%] top-[20%] left-[20%] border-4 border-transparent border-t-[#00D4FF] border-r-[#818CF8] rounded-full animate-[spin_1.5s_infinite]"></div>
                    <div className="absolute w-5 h-5 bg-gradient-to-br from-[#00D4FF] to-[#818CF8] rounded-full top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-[pulse_1.5s_ease-in-out_infinite] shadow-[0_0_20px_rgba(0,212,255,0.4)]"></div>
                </div>

                <div className="text-[1.1rem] font-semibold mb-2 text-[#f0f6fc]">{title}</div>
                <div className="text-[0.85rem] text-[#8b949e] mb-4">{filename}</div>

                <div className="w-full mt-5">
                    <div className="w-full h-1 bg-[#1a2332] rounded-sm overflow-hidden">
                        <div
                            className="h-full bg-gradient-to-r from-[#00D4FF] via-[#818CF8] to-[#00D4FF] bg-[length:200%_100%] rounded-sm transition-all duration-300 ease-out shadow-[0_0_10px_rgba(0,212,255,0.4)] animate-[progress-shimmer_2s_linear_infinite]"
                            style={{ width: `${progress}%` }}
                        ></div>
                    </div>
                </div>
            </div>
        </div>
    );
}
