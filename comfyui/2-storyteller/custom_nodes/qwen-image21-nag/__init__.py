"""Normalized Attention Guidance (NAG) for Qwen Image 2.1.

The core NAGuidance node patches the attention output through attn1_output_patch,
which the Qwen Image 2.1 model does not call, and it needs the positive and the
negative prompt in one batch. Qwen Image 2.1 runs them as separate calls when the
two prompts have different token lengths, which is the normal case.

This node therefore:
1. Takes over the cond/uncond evaluation of each step (sampler_calc_cond_batch_function):
   the negative branch runs first, then the positive branch.
2. Wraps the forward of each attention module. In the negative pass it stores the
   attention output of the target image rows for each block. In the positive pass it
   mixes the stored negative output into the positive output with the NAG formula.

Rows before prefix_len (text and reference) are not changed; with the prefix KV cache,
the target rows come with prefix_len 0, and the prefix rows run with prefix_len equal
to their own length.

NAG makes the negative prompt work at cfg 1, so a distilled model does not overshoot.
The negative branch still runs, so sampling takes approximately 2x time.
"""

import logging

import torch

import comfy.ldm.qwen_image21.model as qwen21
import comfy.samplers

EPS = 1e-6


def nag_mix(z_pos, z_neg, scale, alpha, tau):
    guided = z_pos * scale - z_neg * (scale - 1.0)
    norm_pos = torch.norm(z_pos, p=1, dim=-1, keepdim=True).clamp_min(EPS)
    norm_guided = torch.norm(guided, p=1, dim=-1, keepdim=True).clamp_min(EPS)
    ratio = norm_guided / norm_pos
    guided = guided * (torch.minimum(ratio, torch.full_like(ratio, tau)) / ratio)
    return guided * alpha + z_pos * (1.0 - alpha)


class QwenImage21NAG:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "nag_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 50.0, "step": 0.1,
                                    "tooltip": "How far the attention moves away from the negative prompt. "
                                               "On Qwen Image 2.1 Turbo, 2 and more break the image."}),
            "nag_alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "Blend of the guided attention: 1 is full replacement, 0 is no effect."}),
            "nag_tau": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 10.0, "step": 0.01,
                                  "tooltip": "Limit of the norm ratio between the guided and the positive attention."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/guidance"
    DESCRIPTION = "Normalized Attention Guidance for Qwen Image 2.1: a negative prompt at cfg 1."

    def patch(self, model, nag_scale, nag_alpha, nag_tau):
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")
        state = {"phase": None, "negative": {}, "mixed": 0, "missing": 0}

        def make_forward(module):
            original = qwen21.Attention.forward.__get__(module)

            def forward(x, pe, attn_fn, prefix_len, transformer_options={}):
                out = original(x, pe, attn_fn, prefix_len, transformer_options)
                if state["phase"] is None or prefix_len >= out.shape[1]:
                    return out  # no NAG step, or a prefix-only pass of the KV cache
                block = transformer_options.get("block_index")
                if state["phase"] == "negative":
                    state["negative"][block] = out[:, prefix_len:].detach()
                    return out
                z_neg = state["negative"].get(block)
                if z_neg is None or z_neg.shape != out[:, prefix_len:].shape:
                    state["missing"] += 1
                    return out
                out = out.clone()
                out[:, prefix_len:] = nag_mix(out[:, prefix_len:], z_neg, nag_scale, nag_alpha, nag_tau)
                state["mixed"] += 1
                return out
            return forward

        def calc_cond_batch(args):
            cond, uncond = args["conds"]
            run = lambda c: comfy.samplers.calc_cond_batch(args["model"], [c], args["input"], args["sigma"],
                                                           args["model_options"])[0]
            if uncond is None:
                return [run(cond), None]
            try:
                state["phase"], state["negative"] = "negative", {}
                out_uncond = run(uncond)
                state["phase"] = "positive"
                out_cond = run(cond)
            finally:
                state["phase"], state["negative"] = None, {}
            if not state.get("reported"):
                state["reported"] = True
                level = logging.WARNING if not state["mixed"] else logging.INFO
                logging.log(level, f"QwenImage21NAG: first step mixed {state['mixed']} attention outputs, "
                                   f"{state['missing']} without a negative output")
            return [out_cond, out_uncond]

        count = 0
        for name, module in diffusion_model.named_modules():
            if isinstance(module, qwen21.Attention):
                m.add_object_patch(f"diffusion_model.{name}.forward", make_forward(module))
                count += 1
        if count == 0:
            logging.warning("QwenImage21NAG: no Qwen Image 2.1 attention found; the model is not changed")
        m.model_options["sampler_calc_cond_batch_function"] = calc_cond_batch
        m.disable_model_cfg1_optimization()  # the negative branch must run also at cfg 1
        logging.info(f"QwenImage21NAG: patched {count} attention modules (scale {nag_scale}, alpha {nag_alpha}, tau {nag_tau})")
        return (m,)


NODE_CLASS_MAPPINGS = {"QwenImage21NAG": QwenImage21NAG}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenImage21NAG": "Qwen Image 2.1 NAG"}
