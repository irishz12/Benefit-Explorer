"use client";

import * as AccordionPrimitive from "@radix-ui/react-accordion";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

export const Accordion = AccordionPrimitive.Root;

export function AccordionItem({ className, ...props }: AccordionPrimitive.AccordionItemProps) {
  return <AccordionPrimitive.Item className={cn("border-b border-stone-200", className)} {...props} />;
}

export function AccordionTrigger({ className, children, ...props }: AccordionPrimitive.AccordionTriggerProps) {
  return (
    <AccordionPrimitive.Header>
      <AccordionPrimitive.Trigger
        className={cn("group flex w-full items-center justify-between gap-3 py-4 text-left", className)}
        {...props}
      >
        {children}
        <ChevronDown className="size-4 shrink-0 text-stone-400 transition-transform group-data-[state=open]:rotate-180" />
      </AccordionPrimitive.Trigger>
    </AccordionPrimitive.Header>
  );
}

export function AccordionContent({ className, children, ...props }: AccordionPrimitive.AccordionContentProps) {
  return (
    <AccordionPrimitive.Content className="overflow-hidden data-[state=closed]:animate-accordion-up data-[state=open]:animate-accordion-down" {...props}>
      <div className={cn("pb-4", className)}>{children}</div>
    </AccordionPrimitive.Content>
  );
}
