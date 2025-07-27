/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverActions: {
      allowedOrigins: ["localhost:3000"]
    }
  },
  // Enable SCSS support
  sassOptions: {
    includePaths: ['./src'],
  },
}

module.exports = nextConfig
