// Master effects pass for MUR.
//
// Applies glitch, RGB shift, color grading, strobe/invert, the master dimmer
// and blackout to the composited 1920x1080 output.

struct EffectsUniform {
	rgb_shift: f32,
	glitch: f32,
	strobe: f32,
	strobe_speed_hz: f32,
	brightness: f32,
	contrast: f32,
	saturation: f32,
	master_dimmer: f32,
	time_secs: f32,
	blackout: u32,
	pad: vec2<u32>,
};

struct VsOut {
	@builtin(position) position: vec4<f32>,
	@location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> fx: EffectsUniform;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var input_sampler: sampler;

const TAU: f32 = 6.2831853;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
	let x = f32((index << 1u) & 2u);
	let y = f32(index & 2u);
	var out: VsOut;
	out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
	out.uv = vec2<f32>(x, y);
	return out;
}

fn hash21(p: vec2<f32>) -> f32 {
	let h = dot(p, vec2<f32>(127.1, 311.7));
	return fract(sin(h) * 43758.5453123);
}

fn sample_input(uv: vec2<f32>) -> vec3<f32> {
	return textureSampleLevel(input_tex, input_sampler, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb;
}

fn glitch_offset(uv: vec2<f32>, amount: f32, time: f32) -> vec2<f32> {
	// Discrete time steps make the noise jump instead of sliding smoothly.
	let step_t = floor(time * 14.0);
	var offset = vec2<f32>(0.0);

	// Horizontal row displacement.
	let row = floor(uv.y * 27.0);
	let row_hit = hash21(vec2<f32>(row, step_t));
	if (row_hit < amount * 0.55) {
		offset.x = offset.x + (hash21(vec2<f32>(row + 7.13, step_t + 3.7)) - 0.5) * 0.22 * amount;
	}

	// Rectangular block displacement.
	let cell = floor(uv * vec2<f32>(32.0, 18.0));
	let cell_hit = hash21(cell + vec2<f32>(step_t * 1.37, step_t * 0.71));
	if (cell_hit < amount * 0.18) {
		let jitter = vec2<f32>(hash21(cell + vec2<f32>(3.1, step_t)), hash21(cell + vec2<f32>(step_t, 9.7)));
		offset = offset + (jitter - vec2<f32>(0.5)) * vec2<f32>(0.09, 0.03) * amount;
	}

	// Thin horizontal tear sweeping down the frame.
	let tear_y = fract(time * 0.37);
	let tear_width = 0.004 + 0.02 * amount;
	if (abs(uv.y - tear_y) < tear_width) {
		offset.x = offset.x + 0.035 * amount * sin(time * 53.0 + uv.y * 400.0);
	}
	return offset;
}

@fragment
fn fs_main(vin: VsOut) -> @location(0) vec4<f32> {
	if (fx.blackout != 0u) {
		return vec4<f32>(0.0, 0.0, 0.0, 1.0);
	}

	var uv = vin.uv;
	if (fx.glitch > 0.0) {
		uv = uv + glitch_offset(vin.uv, fx.glitch, fx.time_secs);
	}

	var rgb: vec3<f32>;
	if (fx.rgb_shift > 0.0) {
		let amount = fx.rgb_shift * 0.012;
		let radial = (uv - vec2<f32>(0.5)) * amount * 0.9;
		let shift = vec2<f32>(amount, 0.0) + radial;
		let r = sample_input(uv + shift).r;
		let g = sample_input(uv).g;
		let b = sample_input(uv - shift).b;
		rgb = vec3<f32>(r, g, b);
	} else {
		rgb = sample_input(uv);
	}

	// Color grading.
	rgb = rgb + vec3<f32>(fx.brightness);
	rgb = (rgb - vec3<f32>(0.5)) * fx.contrast + vec3<f32>(0.5);
	let luma = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
	rgb = mix(vec3<f32>(luma), rgb, fx.saturation);
	rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));

	// Strobe / invert.
	if (fx.strobe > 0.0) {
		let on = sin(fx.time_secs * fx.strobe_speed_hz * TAU) >= 0.0;
		if (on) {
			rgb = mix(rgb, vec3<f32>(1.0) - rgb, fx.strobe);
		}
	}

	rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)) * fx.master_dimmer;
	return vec4<f32>(rgb, 1.0);
}
