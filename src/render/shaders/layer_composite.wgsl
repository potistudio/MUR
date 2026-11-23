// Layer compositing pass for MUR.
//
// Reads the previous ping-pong target (`dst_tex`) and one layer texture
// (`src_tex`), fits the layer into its aspect-ratio-preserving rectangle and
// blends it onto the destination with one of five blend modes.

struct LayerUniform {
	// Fitted rectangle in output UV space: (min_u, min_v, max_u, max_v).
	src_rect: vec4<f32>,
	// Right/bottom UV bound of the meaningful texels (crops BC block padding).
	uv_max: vec2<f32>,
	// Layer opacity in 0..1.
	opacity: f32,
	// 0 = Normal, 1 = Add, 2 = Multiply, 3 = Screen, 4 = Difference.
	blend_mode: u32,
	// 1 when the texture holds DXT5-YCoCg (HAP Q) data.
	is_ycocg: u32,
	// 1 when margins outside `src_rect` are opaque black, 0 when transparent.
	opaque_margin: u32,
	// 1 when the texel alpha must be ignored (HAP RGB uploaded as BC1, whose
	// three-color blocks would otherwise decode index 3 as transparent black).
	force_opaque: u32,
	pad: u32,
};

struct VsOut {
	@builtin(position) position: vec4<f32>,
	@location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> layer: LayerUniform;
@group(1) @binding(0) var dst_tex: texture_2d<f32>;
@group(2) @binding(0) var src_tex: texture_2d<f32>;
@group(2) @binding(1) var src_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
	// Single oversized triangle covering the whole viewport.
	let x = f32((index << 1u) & 2u);
	let y = f32(index & 2u);
	var out: VsOut;
	out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
	out.uv = vec2<f32>(x, y);
	return out;
}

fn ycocg_to_rgb(c: vec4<f32>) -> vec3<f32> {
	let scale = (c.b * (255.0 / 8.0)) + 1.0;
	let co = (c.r - 128.0 / 255.0) / scale;
	let cg = (c.g - 128.0 / 255.0) / scale;
	let y = c.a;
	return clamp(vec3<f32>(y + co - cg, y + cg, y - co - cg), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(vin: VsOut) -> @location(0) vec4<f32> {
	let dims = vec2<i32>(textureDimensions(dst_tex));
	let pixel = clamp(vec2<i32>(vin.position.xy), vec2<i32>(0), dims - vec2<i32>(1));
	let dst = textureLoad(dst_tex, pixel, 0).rgb;

	let rect_min = layer.src_rect.xy;
	let rect_max = layer.src_rect.zw;
	let inside = all(vin.uv >= rect_min) && all(vin.uv <= rect_max);

	// Map output UV into the layer texture, keeping half a texel away from the
	// padding so linear filtering never bleeds BC block padding into the edge.
	let extent = max(rect_max - rect_min, vec2<f32>(1e-6));
	let local = clamp((vin.uv - rect_min) / extent, vec2<f32>(0.0), vec2<f32>(1.0));
	let tex_size = vec2<f32>(textureDimensions(src_tex));
	let half_texel = vec2<f32>(0.5) / tex_size;
	let uv_hi = max(layer.uv_max - half_texel, half_texel);
	let tex_uv = clamp(local * layer.uv_max, half_texel, uv_hi);
	let texel = textureSampleLevel(src_tex, src_sampler, tex_uv, 0.0);

	var src_rgb = texel.rgb;
	var src_a = texel.a;
	if (layer.is_ycocg != 0u) {
		src_rgb = ycocg_to_rgb(texel);
		src_a = 1.0;
	} else if (layer.force_opaque != 0u) {
		src_a = 1.0;
	}
	src_a = src_a * layer.opacity;

	if (!inside) {
		src_rgb = vec3<f32>(0.0);
		src_a = select(0.0, layer.opacity, layer.opaque_margin != 0u);
	}
	src_a = clamp(src_a, 0.0, 1.0);

	var out_rgb: vec3<f32>;
	switch layer.blend_mode {
		case 1u: {
			out_rgb = clamp(dst + src_rgb * src_a, vec3<f32>(0.0), vec3<f32>(1.0));
		}
		case 2u: {
			out_rgb = mix(dst, dst * src_rgb, src_a);
		}
		case 3u: {
			out_rgb = mix(dst, vec3<f32>(1.0) - (vec3<f32>(1.0) - dst) * (vec3<f32>(1.0) - src_rgb), src_a);
		}
		case 4u: {
			out_rgb = mix(dst, abs(dst - src_rgb), src_a);
		}
		default: {
			out_rgb = mix(dst, src_rgb, src_a);
		}
	}
	return vec4<f32>(out_rgb, 1.0);
}
