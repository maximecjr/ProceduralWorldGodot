#[compute]
#version 450

// Define the compute shader's local work group size
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Uniform buffer for noise parameters
layout(set = 0, binding = 0, std140) uniform NoiseParams {
    float noise_scale;
    int octaves;
    float persistence;
    float lacunarity;
    int seed;
    vec2 offset;
} noise_params;

// Image to write height map data to
layout(r32f, set = 0, binding = 1) uniform image2D height_map;

// Simple 2D hash function for noise
vec2 hash(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

// Simplex 2D noise
//
vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }

float snoise(vec2 v){
  const vec4 C = vec4(0.211324865405187, 0.366025403784439,
           -0.577350269189626, 0.024390243902439);
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);
  vec2 i1;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod(i, 289.0);
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
  + i.x + vec3(0.0, i1.x, 1.0 ));
  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
    dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;
    float normalization = 0.0;
    
    // Add different octaves of noise
    for (int i = 0; i < noise_params.octaves; i++) {
        // Notice how we use both frequency and amplitude here
        value += amplitude * snoise(p * frequency);
        
        // Track total amplitude for normalization
        normalization += amplitude;
        
        // PROPER USE OF PERSISTENCE: Decreases amplitude with each octave
        amplitude *= noise_params.persistence;
        
        // PROPER USE OF LACUNARITY: Increases frequency with each octave
        frequency *= noise_params.lacunarity;
    }
    
    // Normalize the result to 0-1 range
    value = value / normalization;
    
    // Convert from -1..1 to 0..1 range
    return (value + 1.0) * 0.5;
}

// Main compute shader function
void main() {
    // Get the current pixel coordinate
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

    // Get image dimensions
    ivec2 image_size = imageSize(height_map);
    
    // Check if we're within the image bounds
    if (pixel_coords.x >= image_size.x || pixel_coords.y >= image_size.y) {
        return;
    }

    // Calculate the sample position in world/noise space
    vec2 uv = vec2(pixel_coords) / vec2(image_size);
    vec2 sample_pos = uv * noise_params.noise_scale + noise_params.offset;

    // Apply additional seed offset (just using the seed as a vec2 offset)
    sample_pos += vec2(noise_params.seed * 12.9898, noise_params.seed * 78.233);

    // Generate noise value (range 0 to 1)
    float noise_value = fbm(sample_pos);

    // Write the height value to the image
    imageStore(height_map, pixel_coords, vec4(noise_value, 0.0, 0.0, 1.0));
}