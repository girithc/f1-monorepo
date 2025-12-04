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
import type { PredictResponse } from '@/lib/types';

const chartConfig = {
  probability: {
    label: 'Probability',
    color: 'hsl(var(--primary))',
  },
} satisfies ChartConfig;

type PositionDistributionChartProps = {
  data: PredictResponse['positionProbs'];
};

export function PositionDistributionChart({ data }: PositionDistributionChartProps) {
  const chartData = Object.entries(data)
    .map(([key, value]) => ({
      position: key,
      probability: value,
    }))
    .sort((a, b) => parseInt(a.position.slice(1)) - parseInt(b.position.slice(1)));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Finish Position Probability</CardTitle>
        <CardDescription>
          Predicted probability distribution for each finishing position.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig} className="h-64 w-full">
          <BarChart accessibilityLayer data={chartData}>
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="position"
              tickLine={false}
              tickMargin={10}
              axisLine={false}
              tickFormatter={(value) => value.slice(1)}
            />
            <YAxis 
                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
            />
            <ChartTooltip
              cursor={false}
              content={
                <ChartTooltipContent
                  formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`}
                />
              }
            />
            <Bar dataKey="probability" fill="var(--color-probability)" radius={4} />
          </BarChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
