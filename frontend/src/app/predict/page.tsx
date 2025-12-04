'use client';

import { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { predictRequestSchema } from '@/lib/schemas';
import type { CircuitMeta, PredictResponse } from '@/lib/types';
import { getMetadata, predict } from '@/lib/api-client';
import { PitStopInput } from '@/components/pit-stop-input';
import { PositionDistributionChart } from '@/components/charts/position-distribution-chart';
import ExplanationCard from '@/components/explanation-card';
import { useToast } from '@/hooks/use-toast';
import { Loader2, Compass } from 'lucide-react';

type PredictFormData = z.infer<typeof predictRequestSchema>;

export default function PredictPage() {
  const { toast } = useToast();
  const [circuits, setCircuits] = useState<CircuitMeta[]>([]);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    getMetadata()
      .then((data) => setCircuits(data.circuits))
      .catch(() => toast({
        variant: "destructive",
        title: "Error fetching metadata",
        description: "Could not load circuit data. Please try again later.",
      }));
  }, [toast]);

  const form = useForm<PredictFormData>({
    resolver: zodResolver(predictRequestSchema),
    defaultValues: {
      circuitId: '',
      gridPosition: 10,
      pitPlan: [],
      carPerformanceIndex: 0.5,
      round: 12,
    },
  });

  const onSubmit = async (data: PredictFormData) => {
    setIsLoading(true);
    setPrediction(null);
    try {
      const result = await predict(data);
      setPrediction(result);
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Prediction Failed",
        description: error instanceof Error ? error.message : "An unknown error occurred.",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="grid items-start gap-8 md:grid-cols-2 lg:grid-cols-5">
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle>Prediction Inputs</CardTitle>
          <CardDescription>
            Configure the race scenario to predict the outcome.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
              <FormField
                control={form.control}
                name="circuitId"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Circuit</FormLabel>
                    <Select onValueChange={field.onChange} defaultValue={String(field.value)}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder="Select a circuit" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        {circuits.map((c) => (
                          <SelectItem key={c.circuitId} value={String(c.circuitId)}>
                            {c.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="gridPosition"
                render={({ field }) => (
                  <FormItem>
                    <div className="flex justify-between">
                      <FormLabel>Grid Position</FormLabel>
                      <span className="text-sm text-muted-foreground">{field.value}</span>
                    </div>
                    <FormControl>
                      <Slider
                        min={1}
                        max={20}
                        step={1}
                        value={[field.value]}
                        onValueChange={(vals) => field.onChange(vals[0])}
                      />
                    </FormControl>
                  </FormItem>
                )}
              />

              <PitStopInput control={form.control} />

              <FormField
                control={form.control}
                name="carPerformanceIndex"
                render={({ field }) => (
                  <FormItem>
                    <div className="flex justify-between">
                      <FormLabel>Car Performance Index</FormLabel>
                      <span className="text-sm text-muted-foreground">{field.value.toFixed(2)}</span>
                    </div>
                    <FormControl>
                      <Slider
                        min={0}
                        max={1}
                        step={0.01}
                        value={[field.value]}
                        onValueChange={(vals) => field.onChange(vals[0])}
                      />
                    </FormControl>
                  </FormItem>
                )}
              />



              <Button type="submit" className="w-full" disabled={isLoading}>
                {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                Predict Finish Position
              </Button>
            </form>
          </Form>
        </CardContent>
      </Card>

      <div className="lg:col-span-3 space-y-8">
        {prediction ? (
          <>
            <div className="grid gap-4 md:grid-cols-3">
              <Card>
                <CardHeader>
                  <CardTitle>P50 Finish</CardTitle>
                  <CardDescription>Median</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-4xl font-bold">{prediction.prediction.finishP50.toFixed(1)}</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>Finish Range</CardTitle>
                  <CardDescription>P10 - P90</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-4xl font-bold">
                    {prediction.prediction.finishP10.toFixed(1)} - {prediction.prediction.finishP90.toFixed(1)}
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>Top 3 Probability</CardTitle>
                  <CardDescription>Podium Chance</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-4xl font-bold">
                    {(prediction.top3.probability * 100).toFixed(1)}%
                  </p>
                </CardContent>
              </Card>
            </div>
            <PositionDistributionChart data={prediction.positionProbs} />
            <div className="mt-6">
              <ExplanationCard
                featureImpacts={prediction.explanation?.featureImpacts ?? []}
              />
            </div>
          </>
        ) : (
          <Card className="flex flex-col items-center justify-center min-h-[400px] text-center">
            <CardHeader>
              <CardTitle>Awaiting Prediction</CardTitle>
              <CardDescription>
                {isLoading ? 'Running simulation...' : 'Fill out the form to see the predicted race outcome.'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <Loader2 className="h-12 w-12 animate-spin text-primary" />
              ) : (
                <Compass className="h-12 w-12 text-muted-foreground" />
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
