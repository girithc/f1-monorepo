'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { BarChart3, Bot, Compass, Home, Variable, Activity } from 'lucide-react';

import {
  Sidebar,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
} from '@/components/ui/sidebar';
import { F1RaceStrategyLogo } from '@/components/icons';
import { cn } from '@/lib/utils';

const navItems = [
  { href: '/', label: 'Dashboard', icon: Home },
  { href: '/predict', label: 'Predict', icon: Compass },
  { href: '/compare', label: 'Compare', icon: BarChart3 },

  { href: '/strategy', label: 'LSTM Predictor', icon: Activity },

];

export function SiteSidebar() {
  const pathname = usePathname();

  return (
    <Sidebar className="hidden md:flex md:flex-col md:border-r" collapsible="icon">
      <SidebarHeader className="p-4">
        <Link
          href="/"
          className="flex items-center gap-2 text-primary group-data-[state=collapsed]:justify-center"
        >
          <F1RaceStrategyLogo className="w-8 h-8" />
          <span className="font-headline text-2xl font-semibold group-data-[state=collapsed]:hidden">
            F1 Race Strategy
          </span>
        </Link>
      </SidebarHeader>
      <SidebarMenu className="flex-1 p-2">
        {navItems.map((item) => (
          <SidebarMenuItem key={item.href}>
            <Link href={item.href}>
              <SidebarMenuButton
                variant="ghost"
                isActive={pathname === item.href}
                tooltip={{ children: item.label }}
                className={cn('justify-start', pathname === item.href && 'bg-card')}
              >
                <item.icon />
                <span>{item.label}</span>
              </SidebarMenuButton>
            </Link>
          </SidebarMenuItem>
        ))}
      </SidebarMenu>
    </Sidebar>
  );
}
