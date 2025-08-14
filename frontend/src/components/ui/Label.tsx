import React from "react";

export const Label: React.FC<React.HTMLAttributes<HTMLLabelElement>> = ({ className="", ...props }) => {
  return <label className={`mb-1 block text-sm font-medium text-gray-700 ${className}`} {...props} />;
};
