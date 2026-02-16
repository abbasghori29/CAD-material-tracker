'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { User, ArrowRight, Mail, UserCircle } from 'lucide-react';

import { getApiUrl } from '../../utils/api';

interface UserData {
    id: number;
    email: string;
    full_name: string | null;
    is_active: boolean;
}

export default function ProfilePage() {
    const router = useRouter();
    const [user, setUser] = useState<UserData | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const token = typeof window !== 'undefined' ? localStorage.getItem('ab_builders_token') : null;
        if (!token) {
            router.replace('/login');
            return;
        }

        fetch(`${getApiUrl()}/auth/me`, {
            headers: { Authorization: `Bearer ${token}` },
        })
            .then((res) => {
                if (!res.ok) throw new Error('Unauthorized');
                return res.json();
            })
            .then((data) => {
                setUser(data);
            })
            .catch(() => {
                localStorage.removeItem('ab_builders_token');
                router.replace('/login');
            })
            .finally(() => setLoading(false));
    }, [router]);

    if (loading) {
        return (
            <main className="min-h-screen bg-[#0a0e17] flex items-center justify-center">
                <div className="w-8 h-8 border-2 border-[#00D4FF] border-t-transparent rounded-full animate-spin" />
            </main>
        );
    }

    if (!user) return null;

    const displayName = user.full_name || user.email?.split('@')[0] || 'User';
    const initials = displayName
        .split(' ')
        .map((w) => w[0])
        .join('')
        .toUpperCase()
        .slice(0, 2);

    return (
        <main className="min-h-screen bg-[#0a0e17] text-[#f0f6fc]">
            <div className="fixed inset-0 bg-grid pointer-events-none z-0" aria-hidden />
            <div className="fixed -top-40 -right-40 w-96 h-96 rounded-full bg-[#00D4FF]/5 blur-[120px] pointer-events-none z-0" aria-hidden />
            <div className="fixed -bottom-40 -left-40 w-96 h-96 rounded-full bg-[#818CF8]/5 blur-[120px] pointer-events-none z-0" aria-hidden />

            <div className="relative z-10 max-w-2xl mx-auto px-6 py-12 pb-24">
                <Link
                    href="/"
                    className="inline-flex items-center gap-2 text-[#8b949e] hover:text-[#00D4FF] text-sm font-medium mb-10 transition-colors"
                >
                    <ArrowRight className="w-4 h-4 rotate-180" /> Back to Dashboard
                </Link>

                <header className="mb-10">
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-[#00D4FF] to-[#818CF8] bg-clip-text text-transparent">
                        Profile
                    </h1>
                    <p className="text-[#8b949e] mt-1">Your account details</p>
                </header>

                <div className="bg-[#111827] border border-[#30363d] rounded-2xl overflow-hidden shadow-[0_20px_60px_rgba(0,0,0,0.4)]">
                    <div className="p-8 border-b border-[#30363d] flex items-center gap-5">
                        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#00D4FF]/20 to-[#818CF8]/20 border border-[#00D4FF]/30 flex items-center justify-center flex-shrink-0">
                            <span className="text-2xl font-bold bg-gradient-to-br from-[#00D4FF] to-[#818CF8] bg-clip-text text-transparent">
                                {initials}
                            </span>
                        </div>
                        <div>
                            <p className="text-xl font-semibold text-white">{displayName}</p>
                            <p className="text-[#8b949e] text-sm mt-0.5">{user.email}</p>
                        </div>
                    </div>

                    <div className="divide-y divide-[#30363d]">
                        <div className="flex items-center gap-4 px-8 py-4">
                            <UserCircle className="w-5 h-5 text-[#00D4FF] flex-shrink-0" />
                            <div className="min-w-0">
                                <p className="text-xs text-[#8b949e] uppercase tracking-wider font-medium">Full name</p>
                                <p className="text-[#f0f6fc] font-medium truncate">{user.full_name || '—'}</p>
                            </div>
                        </div>
                        <div className="flex items-center gap-4 px-8 py-4">
                            <Mail className="w-5 h-5 text-[#00D4FF] flex-shrink-0" />
                            <div className="min-w-0">
                                <p className="text-xs text-[#8b949e] uppercase tracking-wider font-medium">Email</p>
                                <p className="text-[#f0f6fc] font-medium truncate">{user.email}</p>
                            </div>
                        </div>
                        <div className="flex items-center gap-4 px-8 py-4">
                            <User className="w-5 h-5 text-[#00D4FF] flex-shrink-0" />
                            <div className="min-w-0">
                                <p className="text-xs text-[#8b949e] uppercase tracking-wider font-medium">Status</p>
                                <p className="text-[#f0f6fc] font-medium">
                                    {user.is_active ? (
                                        <span className="text-[#10B981]">Active</span>
                                    ) : (
                                        <span className="text-[#8b949e]">Inactive</span>
                                    )}
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    );
}
