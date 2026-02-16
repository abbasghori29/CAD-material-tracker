'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Building2, Eye, EyeOff } from 'lucide-react';

import { getApiUrl } from '../../utils/api';

export default function LoginPage() {
    const router = useRouter();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError(null);
        setLoading(true);

        try {
            const res = await fetch(`${getApiUrl()}/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password }),
            });

            if (!res.ok) {
                const data = await res.json().catch(() => ({}));
                throw new Error(data.detail || 'Invalid credentials');
            }

            const data = await res.json();
            const token = data.access_token as string;
            if (token) {
                localStorage.setItem('ab_builders_token', token);
                router.push('/');
            } else {
                throw new Error('No token returned from server');
            }
        } catch (err: any) {
            setError(err.message || 'Login failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <main className="min-h-screen bg-[#0a0e17] text-[#f0f6fc] flex items-center justify-center px-4">
            {/* Subtle background grid */}
            <div className="bg-grid" />

            <div className="relative z-10 w-full max-w-[1000px] flex rounded-[24px] overflow-hidden shadow-[0_40px_100px_rgba(0,0,0,0.7)] border border-[#1e293b]/60">

                {/* ─── Left brand panel ─── */}
                <div className="hidden md:flex md:w-[45%] relative flex-col justify-between bg-gradient-to-br from-[#0c1629] via-[#111d35] to-[#0a1628] p-10 overflow-hidden">
                    {/* Decorative gradient orbs */}
                    <div className="absolute -top-20 -left-20 w-72 h-72 rounded-full bg-[#00D4FF]/10 blur-[100px] pointer-events-none" />
                    <div className="absolute -bottom-16 -right-16 w-60 h-60 rounded-full bg-[#818CF8]/10 blur-[80px] pointer-events-none" />

                    {/* Decorative grid lines (like CAD blueprint) */}
                    <div className="absolute inset-0 opacity-[0.04]" style={{
                        backgroundImage: 'linear-gradient(#00D4FF 1px, transparent 1px), linear-gradient(90deg, #00D4FF 1px, transparent 1px)',
                        backgroundSize: '40px 40px',
                    }} />

                    {/* Top: logo */}
                    <div className="relative z-10">
                        <div className="w-14 h-14 rounded-2xl flex items-center justify-center shadow-[0_4px_30px_rgba(0,212,255,0.3)] bg-gradient-to-br from-[#00D4FF] to-[#818CF8] mb-8">
                            <Building2 className="w-7 h-7 text-black" />
                        </div>
                    </div>

                    {/* Middle: hero text */}
                    <div className="relative z-10 flex-1 flex flex-col justify-center -mt-6">
                        <p className="text-sm font-medium tracking-widest uppercase text-[#00D4FF]/70 mb-3">
                            Welcome to
                        </p>
                        <h1 className="text-[2.4rem] leading-[1.15] font-bold tracking-tight">
                            <span className="bg-gradient-to-r from-[#00D4FF] to-[#818CF8] bg-clip-text text-transparent">
                                Absolute
                            </span>
                            <br />
                            <span className="text-white">
                                Builders
                            </span>
                        </h1>
                        <p className="mt-5 text-[15px] leading-relaxed text-[#8b949e] max-w-[280px]">
                            AI-powered CAD material tracking and estimation, purpose-built for your engineering team.
                        </p>
                    </div>

                    {/* Bottom: copyright */}
                    <p className="relative z-10 text-xs text-[#8b949e]/50">
                        © {new Date().getFullYear()} Absolute Builders
                    </p>
                </div>

                {/* ─── Right login form ─── */}
                <div className="w-full md:w-[55%] bg-[#111827] px-8 sm:px-12 py-12 flex flex-col justify-center">
                    {/* Mobile-only logo */}
                    <div className="md:hidden flex items-center gap-3 mb-8">
                        <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-gradient-to-br from-[#00D4FF] to-[#818CF8]">
                            <Building2 className="w-5 h-5 text-black" />
                        </div>
                        <span className="text-lg font-bold bg-gradient-to-r from-[#00D4FF] to-[#818CF8] bg-clip-text text-transparent">
                            Absolute Builders
                        </span>
                    </div>

                    <h2 className="text-[1.65rem] font-bold text-white mb-1">
                        Welcome Back!
                    </h2>
                    <p className="text-sm text-[#8b949e] mb-8">
                        Sign in to your account to continue
                    </p>

                    {error && (
                        <div className="mb-5 rounded-xl border border-red-500/40 bg-red-500/10 px-4 py-2.5 text-sm text-red-300 flex items-center gap-2">
                            <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.28 7.22a.75.75 0 00-1.06 1.06L8.94 10l-1.72 1.72a.75.75 0 101.06 1.06L10 11.06l1.72 1.72a.75.75 0 101.06-1.06L11.06 10l1.72-1.72a.75.75 0 00-1.06-1.06L10 8.94 8.28 7.22z" clipRule="evenodd" />
                            </svg>
                            {error}
                        </div>
                    )}

                    <form onSubmit={handleSubmit} className="space-y-5">
                        {/* Email */}
                        <div>
                            <label className="block text-xs font-medium text-[#8b949e] mb-2 uppercase tracking-wider">
                                Email
                            </label>
                            <input
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                placeholder="you@absolutebuilders.com"
                                className="w-full h-12 rounded-xl bg-[#1a2332] border border-[#30363d] px-4 text-[15px] text-white placeholder:text-[#4b5563] focus:outline-none focus:border-[#00D4FF] focus:shadow-[0_0_0_3px_rgba(0,212,255,0.15)] transition-all"
                                required
                            />
                        </div>

                        {/* Password */}
                        <div>
                            <label className="block text-xs font-medium text-[#8b949e] mb-2 uppercase tracking-wider">
                                Password
                            </label>
                            <div className="relative">
                                <input
                                    type={showPassword ? 'text' : 'password'}
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    placeholder="Enter your password"
                                    className="w-full h-12 rounded-xl bg-[#1a2332] border border-[#30363d] px-4 pr-12 text-[15px] text-white placeholder:text-[#4b5563] focus:outline-none focus:border-[#00D4FF] focus:shadow-[0_0_0_3px_rgba(0,212,255,0.15)] transition-all"
                                    required
                                />
                                <button
                                    type="button"
                                    onClick={() => setShowPassword(!showPassword)}
                                    className="absolute right-4 top-1/2 -translate-y-1/2 text-[#8b949e] hover:text-[#00D4FF] transition-colors"
                                    tabIndex={-1}
                                >
                                    {showPassword
                                        ? <EyeOff className="w-[18px] h-[18px]" />
                                        : <Eye className="w-[18px] h-[18px]" />
                                    }
                                </button>
                            </div>
                        </div>

                        {/* Submit */}
                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full h-12 mt-1 rounded-xl bg-gradient-to-r from-[#00D4FF] to-[#818CF8] text-[#0a0e17] font-semibold text-[15px] shadow-[0_8px_30px_rgba(0,212,255,0.3)] hover:shadow-[0_12px_40px_rgba(0,212,255,0.45)] transition-all hover:-translate-y-0.5 active:translate-y-0 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0 disabled:hover:shadow-[0_8px_30px_rgba(0,212,255,0.3)]"
                        >
                            {loading ? (
                                <span className="flex items-center justify-center gap-2">
                                    <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                    </svg>
                                    Signing In...
                                </span>
                            ) : 'Sign In'}
                        </button>
                    </form>

                    {/* Footer */}
                    <p className="mt-8 text-xs text-[#8b949e]/40 text-center md:hidden">
                        © {new Date().getFullYear()} Absolute Builders. All rights reserved.
                    </p>
                </div>
            </div>
        </main>
    );
}
