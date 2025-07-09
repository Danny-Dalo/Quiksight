import type { Config } from "tailwindcss";

const config: Config = {
  content: ["app_quiksight/templates/**/*.{html,js}"],
  theme: {
    extend: {},
  },
  plugins: [require("daisyui")],
};

export default config;
