import React from "react";

export const Textarea: React.FC<React.TextareaHTMLAttributes<HTMLTextAreaElement>> = ({ className="", ...props }) => {
  return (
    <textarea
      className={`block w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-gray-900 ${className}`}
      {...props}
    />
  );
};
