import type { Metadata } from 'next';
import './globals.css';
import { SidebarProvider } from '@/components/ui/sidebar';
import { SiteSidebar } from '@/components/site-sidebar';
import { SiteHeader } from '@/components/site-header';
import { Toaster } from '@/components/ui/toaster';

export const metadata: Metadata = {
  title: 'F1 Race Strategy',
  description: 'AI-Powered Racing Strategy',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
      </head>
      <body className="font-body antialiased">
        <Toaster />
        <SidebarProvider>
          <div className="flex min-h-screen w-full">
            <SiteSidebar />
            <div className="flex flex-col flex-1">
              <SiteHeader />
              <main className="flex-1 p-4 md:p-6 lg:p-8">
                {children}
              </main>
            </div>
          </div>
        </SidebarProvider>
      </body>
    </html>
  );
}
