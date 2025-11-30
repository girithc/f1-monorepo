import Image from 'next/image';
import Link from 'next/link';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { getMetadata, getHealth } from '@/lib/api-client';
import { PlaceHolderImages } from '@/lib/placeholder-images';
import { Button } from '@/components/ui/button';
import { ArrowRight, BarChart3, Bot, Compass, Variable, Activity } from 'lucide-react';
import type { HealthzResponse, MetadataResponse } from '@/lib/types';

export default async function DashboardPage() {
  const [health, metadata] = await Promise.all([
    getHealth().catch(() => null),
    getMetadata().catch(() => null),
  ]);

  const heroImage = PlaceHolderImages.find((img) => img.id === 'hero-race-car');

  return (
    <div className="space-y-8">
      <Card className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-card/80 via-card/50 to-transparent" />
        {heroImage && (
          <Image
            src={heroImage.imageUrl}
            alt={heroImage.description}
            fill
            className="object-cover"
            data-ai-hint={heroImage.imageHint}
            priority
          />
        )}
        <div className="relative p-8 md:p-12">
          <h1 className="text-4xl md:text-5xl font-bold text-primary font-headline">
            Welcome to F1 Race Strategy
          </h1>
          <p className="mt-2 text-lg text-foreground/90 max-w-2xl">
            Your AI-powered co-pilot for elite Formula 1 racing strategy.
            Analyze scenarios, compare outcomes, and get intelligent pit stop
            advice to gain the ultimate competitive advantage.
          </p>
          <Button asChild className="mt-6">
            <Link href="/predict">
              Start Predicting <ArrowRight className="ml-2" />
            </Link>
          </Button>
        </div>
      </Card>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Model Status</CardTitle>
            <div className={`h-3 w-3 rounded-full ${health?.status === 'ok' ? 'bg-accent' : 'bg-destructive'}`} />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{health?.status === 'ok' ? 'Online' : 'Offline'}</div>
            <p className="text-xs text-muted-foreground">
              Version: {health?.modelVersion || 'N/A'}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Available Circuits</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metadata?.circuits.length || '...'}
            </div>
            <p className="text-xs text-muted-foreground">
              Ready for simulation
            </p>
          </CardContent>
        </Card>
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Core Features</CardTitle>
            <CardDescription>
              Leverage powerful tools to refine your race strategy.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-4 text-sm">
            <Link href="/predict" className="flex items-center gap-2 hover:text-primary">
              <Compass size={16} /> Predict Finish
            </Link>
            <Link href="/compare" className="flex items-center gap-2 hover:text-primary">
              <BarChart3 size={16} /> Compare Strategies
            </Link>
            <Link href="/whatif" className="flex items-center gap-2 hover:text-primary">
              <Variable size={16} /> What-If Analysis
            </Link>
            <Link href="/advisor" className="flex items-center gap-2 hover:text-primary">
              <Bot size={16} /> AI Advisor
            </Link>
            <Link href="/strategy" className="flex items-center gap-2 hover:text-primary">
              <Activity size={16} /> In-Race LSTM Predictor
            </Link>
          </CardContent>

        </Card>
      </div>
    </div>
  );
}
