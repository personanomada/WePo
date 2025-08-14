import React from "react";

export const Input: React.FC<React.InputHTMLAttributes<HTMLInputElement>> = ({ className="", ...props }) => {
  return (
    <input
      className={`block w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-gray-900 ${className}`}
      {...props}
    />
  );
};
