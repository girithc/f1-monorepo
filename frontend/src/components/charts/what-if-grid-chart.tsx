'use client';

import { CartesianGrid, Line, LineChart, XAxis, YAxis } from 'recharts';
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
import type { WhatIfResponse } from '@/lib/types';

const chartConfig = {
  finishP50: {
    label: 'Median Finish',
    color: 'hsl(var(--primary))',
  },
} satisfies ChartConfig;

type WhatIfGridChartProps = {
  data: WhatIfResponse['series'];
};

export function WhatIfGridChart({ data }: WhatIfGridChartProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Grid Position Sensitivity</CardTitle>
        <CardDescription>
          Predicted median finish position based on starting grid position.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig} className="h-80 w-full">
          <LineChart
            accessibilityLayer
            data={data}
            margin={{
              left: 12,
              right: 12,
            }}
          >
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="grid"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              tickFormatter={(value) => `P${value}`}
            />
            <YAxis
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              reversed
              domain={['dataMin - 1', 'dataMax + 1']}
            />
            <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
            <Line
              dataKey="finishP50"
              type="monotone"
              stroke="var(--color-finishP50)"
              strokeWidth={2}
              dot={true}
            />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
