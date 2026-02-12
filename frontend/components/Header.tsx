'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { Building2, LogOut, ChevronDown, User, BookOpen } from 'lucide-react';

interface HeaderProps {
    userName?: string | null;
    userEmail?: string | null;
    onLogout?: () => void;
}

export default function Header({ userName, userEmail, onLogout }: HeaderProps) {
    const [menuOpen, setMenuOpen] = useState(false);

    const displayName = userName || userEmail?.split('@')[0] || 'User';
    const initials = displayName
        .split(' ')
        .map(w => w[0])
        .join('')
        .toUpperCase()
        .slice(0, 2);

    return (
        <header className="relative z-50 flex items-center justify-between px-6 py-4 border-b border-white/[0.06] bg-[#111827]/80 backdrop-blur-xl">
            {/* Left: Logo & branding */}
            <div className="flex items-center gap-4">
                <div className="relative group">
                    <div className="absolute -inset-1 bg-gradient-to-br from-[#00D4FF] to-[#818CF8] rounded-2xl blur-md opacity-40 group-hover:opacity-60 transition-opacity" />
                    <div className="relative w-12 h-12 rounded-xl flex items-center justify-center bg-gradient-to-br from-[#00D4FF] to-[#818CF8] shadow-lg">
                        <Building2 className="w-[26px] h-[26px] text-black" />
                    </div>
                </div>
                <div className="flex flex-col">
                    <span className="text-[1.3rem] font-bold leading-tight bg-gradient-to-r from-[#00D4FF] to-[#818CF8] bg-clip-text text-transparent">
                        Absolute Builders
                    </span>
                    <span className="text-[0.75rem] tracking-wider uppercase text-[#8b949e] font-medium">
                        AI-Powered Estimation Suite
                    </span>
                </div>
            </div>

            {/* Right: User menu */}
            {userName !== undefined && (
                <div className="relative">
                    <button
                        onClick={() => setMenuOpen(!menuOpen)}
                        className="flex items-center gap-3 pl-2 pr-3 py-1.5 rounded-2xl border border-[#30363d] bg-[#1a2332]/60 hover:bg-[#1a2332] hover:border-[#00D4FF]/30 transition-all group"
                    >
                        {/* Avatar */}
                        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-[#00D4FF]/20 to-[#818CF8]/20 border border-[#00D4FF]/20 flex items-center justify-center">
                            <span className="text-xs font-bold bg-gradient-to-br from-[#00D4FF] to-[#818CF8] bg-clip-text text-transparent">
                                {initials}
                            </span>
                        </div>

                        <div className="hidden sm:flex flex-col items-start">
                            <span className="text-sm font-medium text-white leading-tight">
                                Hello, {displayName.split(' ')[0]}
                            </span>
                            {userEmail && (
                                <span className="text-[10px] text-[#8b949e] leading-tight">
                                    {userEmail}
                                </span>
                            )}
                        </div>
                        <ChevronDown className={`w-4 h-4 text-[#8b949e] group-hover:text-[#00D4FF] transition-all ${menuOpen ? 'rotate-180' : ''}`} />
                    </button>

                    {/* Dropdown */}
                    {menuOpen && (
                        <>
                            {/* Invisible backdrop to close menu */}
                            <div className="fixed inset-0 z-40" onClick={() => setMenuOpen(false)} />

                            <div className="absolute right-0 top-[calc(100%+8px)] z-50 w-64 rounded-2xl bg-[#111827] border border-[#30363d] shadow-[0_20px_60px_rgba(0,0,0,0.5)] overflow-hidden animate-in fade-in slide-in-from-top-2 duration-200">
                                {/* User info */}
                                <div className="px-4 py-4 border-b border-[#30363d]/60">
                                    <div className="flex items-center gap-3">
                                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#00D4FF]/20 to-[#818CF8]/20 border border-[#00D4FF]/20 flex items-center justify-center flex-shrink-0">
                                            <span className="text-sm font-bold bg-gradient-to-br from-[#00D4FF] to-[#818CF8] bg-clip-text text-transparent">
                                                {initials}
                                            </span>
                                        </div>
                                        <div className="min-w-0">
                                            <p className="text-sm font-semibold text-white truncate">{displayName}</p>
                                            {userEmail && (
                                                <p className="text-[11px] text-[#8b949e] truncate">{userEmail}</p>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                {/* Menu items */}
                                <div className="p-1.5">
                                    <Link
                                        href="/docs"
                                        onClick={() => setMenuOpen(false)}
                                        className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm text-[#8b949e] hover:text-white hover:bg-[#1a2332] transition-all"
                                    >
                                        <BookOpen className="w-4 h-4" />
                                        Documentation
                                    </Link>
                                    <Link
                                        href="/profile"
                                        onClick={() => setMenuOpen(false)}
                                        className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm text-[#8b949e] hover:text-white hover:bg-[#1a2332] transition-all"
                                    >
                                        <User className="w-4 h-4" />
                                        Profile
                                    </Link>
                                    <button
                                        onClick={() => {
                                            setMenuOpen(false);
                                            onLogout?.();
                                        }}
                                        className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm text-red-400 hover:text-red-300 hover:bg-red-500/10 transition-all"
                                    >
                                        <LogOut className="w-4 h-4" />
                                        Sign Out
                                    </button>
                                </div>
                            </div>
                        </>
                    )}
                </div>
            )}
        </header>
    );
}
