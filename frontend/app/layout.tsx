import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BenefitExplorer — Insurance Answers Grounded in Brochures",
  description: "Grounded answers from insurance product brochures with verified citations.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
