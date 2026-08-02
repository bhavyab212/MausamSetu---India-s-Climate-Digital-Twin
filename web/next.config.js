/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: "/_kitchen-sink",
        destination: "/kitchen-sink",
      },
      {
        source: "/_tokens-test",
        destination: "/tokens-test",
      },
    ]
  },
}

module.exports = nextConfig
