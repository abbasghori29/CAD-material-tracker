import React from 'react';

export default function Footer() {
    return (
        <footer className="relative z-10 border-t border-white/[0.04] bg-[#0a0e17]/80 backdrop-blur-sm">
            <div className="max-w-7xl mx-auto px-6 py-5 flex items-center justify-center">
                <span className="text-xs text-[#8b949e]">
                    © 2026 <span className="text-[#f0f6fc]/70 font-medium">Absolute Builders</span>. All rights reserved.
                </span>
            </div>
        </footer>
    );
}
