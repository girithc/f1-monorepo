'use client';

import { usePathname } from 'next/navigation';
import { Menu, CircleDot } from 'lucide-react';
import { useState, useEffect } from 'react';

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet';
import Link from 'next/link';
import { F1RaceStrategyLogo } from './icons';
import { getHealth } from '@/lib/api-client';

const navItems = [
    { href: '/', label: 'Dashboard' },
    { href: '/predict', label: 'Predict' },
    { href: '/compare', label: 'Compare' },
    { href: '/whatif', label: 'What-If' },
    { href: '/advisor', label: 'AI Advisor' },
];

const pageTitles: { [key: string]: string } = {
  '/': 'Dashboard',
  '/predict': 'Single Scenario Prediction',
  '/compare': 'Strategy Comparison',
  '/whatif': 'Sensitivity Analysis',
  '/advisor': 'AI-Powered Pit Strategy Advisor',
};

export function SiteHeader() {
  const pathname = usePathname();
  const [healthStatus, setHealthStatus] = useState<{ status: 'ok' | 'error', modelVersion: string } | null>(null);

  useEffect(() => {
    getHealth().then(setHealthStatus).catch(() => setHealthStatus({ status: 'error', modelVersion: 'N/A' }));
  }, []);

  return (
    <header className="sticky top-0 z-10 flex h-16 items-center gap-4 border-b bg-background/80 px-4 backdrop-blur-sm md:px-6">
      <nav className="md:hidden">
        <Sheet>
          <SheetTrigger>
            <Menu className="h-6 w-6" />
            <span className="sr-only">Toggle navigation menu</span>
          </SheetTrigger>
          <SheetContent side="left">
            <SheetHeader>
              <SheetTitle>
                <Link href="/" className="flex items-center gap-2 text-primary">
                  <F1RaceStrategyLogo className="w-8 h-8" />
                  <span className="font-headline text-2xl font-semibold">F1 Race Strategy</span>
                </Link>
              </SheetTitle>
              <SheetDescription>
                Navigate through the F1 Race Strategy application features.
              </SheetDescription>
            </SheetHeader>
            <div className="flex flex-col gap-2 py-4">
              {navItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center gap-3 rounded-lg px-3 py-2 text-muted-foreground transition-all hover:text-primary ${pathname === item.href ? 'text-primary bg-muted' : ''}`}
                >
                  {item.label}
                </Link>
              ))}
            </div>
          </SheetContent>
        </Sheet>
      </nav>
      <div className="flex w-full items-center justify-between">
        <h1 className="text-xl font-semibold md:text-2xl font-headline">
          {pageTitles[pathname] || 'F1 Race Strategy'}
        </h1>
        {healthStatus && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground" title={`Model version: ${healthStatus.modelVersion}`}>
            <CircleDot className={healthStatus.status === 'ok' ? 'text-accent' : 'text-destructive'} size={16} />
            <span>{healthStatus.status === 'ok' ? 'Model Online' : 'Model Offline'}</span>
          </div>
        )}
      </div>
    </header>
  );
}
