import React from "react";

export const Button: React.FC<React.ButtonHTMLAttributes<HTMLButtonElement>> = ({ className="", ...props }) => {
  return (
    <button
      className={`inline-flex items-center justify-center rounded-md bg-gray-900 text-white px-4 py-2 text-sm font-medium hover:opacity-90 disabled:opacity-50 ${className}`}
      {...props}
    />
  );
};
