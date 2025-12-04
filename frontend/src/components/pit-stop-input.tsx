'use client';

import { useFieldArray, type Control } from 'react-hook-form';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { PlusCircle, Trash2 } from 'lucide-react';
import type { PredictRequest } from '@/lib/types';
import { FormControl, FormField, FormItem, FormMessage } from './ui/form';

type PitStopInputProps = {
  control: Control<PredictRequest>;
};

export function PitStopInput({ control }: PitStopInputProps) {
  const { fields, append, remove } = useFieldArray({
    control,
    name: 'pitPlan',
  });

  return (
    <div className="space-y-4">
      <div className='flex items-center justify-between'>
        <Label>Pit Plan</Label>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={() => append({ lap: 0, durationMs: 23000 })}
        >
          <PlusCircle className="mr-2 h-4 w-4" />
          Add Pit Stop
        </Button>
      </div>

      {fields.map((field, index) => (
        <div key={field.id} className="flex items-start gap-2">
          <FormField
            control={control}
            name={`pitPlan.${index}.lap`}
            render={({ field }) => (
              <FormItem className="flex-1">
                <Label className="sr-only">Lap</Label>
                <FormControl>
                  <Input type="number" placeholder="Lap" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={control}
            name={`pitPlan.${index}.durationMs`}
            render={({ field }) => (
              <FormItem className="flex-1">
                <Label className="sr-only">Duration (ms)</Label>
                <FormControl>
                  <Input type="number" placeholder="Duration (ms)" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={() => remove(index)}
            className="shrink-0 mt-1"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      ))}
      {fields.length === 0 && (
        <p className="text-sm text-muted-foreground text-center py-2">No pit stops planned.</p>
      )}
    </div>
  );
}
