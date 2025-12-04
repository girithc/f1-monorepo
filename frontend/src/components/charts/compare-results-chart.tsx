'use client';

import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from 'recharts';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartConfig,
} from '@/components/ui/chart';
import type { CompareResult } from '@/lib/types';
import { Badge } from '../ui/badge';

const chartConfig = {
  finishP50: {
    label: 'Median Finish',
    color: 'hsl(var(--chart-1))',
  },
  robustnessScore: {
    label: 'Robustness',
    color: 'hsl(var(--chart-2))',
  },
} satisfies ChartConfig;

type CompareResultsChartProps = {
  results: CompareResult[];
  recommendedScenarioId: string;
};

export function CompareResultsChart({ results, recommendedScenarioId }: CompareResultsChartProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Strategy Comparison</CardTitle>
        <CardDescription>
          Lower median finish and higher robustness score are better.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig} className="h-80 w-full">
          <BarChart accessibilityLayer data={results}>
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="scenarioId"
              tickLine={false}
              tickMargin={10}
              axisLine={false}
              tick={({ x, y, payload }) => (
                <g transform={`translate(${x},${y})`}>
                  <text x={0} y={0} dy={16} textAnchor="middle" fill="hsl(var(--foreground))" fontSize={12}>
                    {payload.value}
                    {payload.value === recommendedScenarioId && (
                      <tspan x={0} dy="1.2em" fontSize={10} fill="hsl(var(--accent))">
                        (Recommended)
                      </tspan>
                    )}
                  </text>
                </g>
              )}
            />
            <YAxis yAxisId="left" stroke="hsl(var(--chart-1))" />
            <YAxis yAxisId="right" orientation="right" stroke="hsl(var(--chart-2))" />
            <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
            <Bar yAxisId="left" dataKey="finishP50" fill="var(--color-finishP50)" radius={4} />
            <Bar yAxisId="right" dataKey="robustnessScore" fill="var(--color-robustnessScore)" radius={4} />
          </BarChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
