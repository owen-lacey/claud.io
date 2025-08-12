import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FPL Team Picker",
  description: "Fantasy Premier League team picker and assistant",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
