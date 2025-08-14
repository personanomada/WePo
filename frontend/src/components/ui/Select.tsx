import React from "react";

export const Select: React.FC<{value:string; onChange:(v:string)=>void; options:{value:string; label:string;}[]}> = ({value, onChange, options}) => {
  return (
    <select
      className="block w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-gray-900"
      value={value}
      onChange={(e)=>onChange(e.target.value)}
    >
      {options.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
    </select>
  );
};
