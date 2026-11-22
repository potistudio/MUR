// Presentation blit for MUR.
//
// Scales the 1920x1080 master output into an aspect-ratio-preserving
// rectangle of the target surface, filling the rest with opaque black.

struct PresentUniform {
	// Fitted rectangle in target UV space: (min_u, min_v, max_u, max_v).
	dst_rect: vec4<f32>,
	// 1 when the target format is sRGB and expects linear values.
	to_linear: u32,
	pad0: u32,
	pad1: u32,
	pad2: u32,
};

struct VsOut {
	@builtin(position) position: vec4<f32>,
	@location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> present: PresentUniform;
@group(0) @binding(1) var master_tex: texture_2d<f32>;
@group(0) @binding(2) var master_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
	let x = f32((index << 1u) & 2u);
	let y = f32(index & 2u);
	var out: VsOut;
	out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
	out.uv = vec2<f32>(x, y);
	return out;
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
	let low = c / 12.92;
	let high = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
	return select(high, low, c <= vec3<f32>(0.04045));
}

@fragment
fn fs_main(vin: VsOut) -> @location(0) vec4<f32> {
	let rect_min = present.dst_rect.xy;
	let rect_max = present.dst_rect.zw;
	let extent = max(rect_max - rect_min, vec2<f32>(1e-6));
	let local = clamp((vin.uv - rect_min) / extent, vec2<f32>(0.0), vec2<f32>(1.0));
	let inside = all(vin.uv >= rect_min) && all(vin.uv <= rect_max);
	var rgb = textureSampleLevel(master_tex, master_sampler, local, 0.0).rgb;
	rgb = select(vec3<f32>(0.0), rgb, inside);
	if (present.to_linear != 0u) {
		rgb = srgb_to_linear(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)));
	}
	return vec4<f32>(rgb, 1.0);
}
