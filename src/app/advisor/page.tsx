'use client';

import { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { getMetadata } from '@/lib/api-client';
import type { CircuitMeta } from '@/lib/types';
import { getAIAdvice } from '@/lib/actions';
import { useToast } from '@/hooks/use-toast';
import { AdvisorForm } from './advisor-form';
import type { SuggestPitStrategyOutput } from '@/ai/flows/ai-pit-strategy-advisor';
import { Bot, Loader2 } from 'lucide-react';

export default function AdvisorPage() {
  const [circuits, setCircuits] = useState<CircuitMeta[]>([]);
  const [advice, setAdvice] = useState<SuggestPitStrategyOutput | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    getMetadata()
      .then((data) => setCircuits(data.circuits))
      .catch(() =>
        toast({
          variant: 'destructive',
          title: 'Error fetching metadata',
          description: 'Could not load circuit data. Please try again later.',
        })
      );
  }, [toast]);

  const handleGetAdvice = async (data: any) => {
    const result = await getAIAdvice(data);
    if (!result.success) {
      toast({
        variant: 'destructive',
        title: 'AI Advisor Failed',
        description: result.error,
      });
    }
    return result;
  }

  return (
    <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-5">
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle>AI Pit Strategy Advisor</CardTitle>
          <CardDescription>
            Get an AI-powered pit strategy suggestion based on race conditions.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {circuits.length > 0 ? (
            <AdvisorForm
              circuits={circuits}
              onGetAdvice={handleGetAdvice}
              setAdvice={setAdvice}
              setIsLoading={setIsLoading}
              isLoading={isLoading}
            />
          ) : (
            <div className="flex items-center justify-center h-40">
              <Loader2 className="h-8 w-8 animate-spin" />
            </div>
          )}
        </CardContent>
      </Card>
      <div className="lg:col-span-3">
        {advice ? (
          <Card>
            <CardHeader>
              <CardTitle>AI Recommended Strategy</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="font-semibold mb-2">Pit Plan</h3>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Pit Stop</TableHead>
                      <TableHead>Lap</TableHead>
                      <TableHead className="text-right">Duration (s)</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {advice.pitPlan.map((stop, index) => (
                      <TableRow key={index}>
                        <TableCell>Stop {index + 1}</TableCell>
                        <TableCell>{stop.lap}</TableCell>
                        <TableCell className="text-right">
                          {(stop.durationMs / 1000).toFixed(2)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
              <div>
                <h3 className="font-semibold mb-2">Explanation</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {advice.explanation}
                </p>
              </div>
            </CardContent>
          </Card>
        ) : (
          <Card className="flex flex-col items-center justify-center min-h-[400px] text-center">
            <CardHeader>
              <CardTitle>Awaiting AI Advice</CardTitle>
              <CardDescription>
                {isLoading ? 'The AI is analyzing the data...' : 'Fill out the form to get a strategy.'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <Loader2 className="h-12 w-12 animate-spin text-primary" />
              ) : (
                <Bot className="h-12 w-12 text-muted-foreground" />
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
