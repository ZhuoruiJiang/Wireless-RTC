"""Lazy headless Kinetix rasterization and JPEG encoding for the canvas viewer."""
import io


class KinetixFrameRenderer:
    def __init__(self, simulator):
        self._simulator = simulator
        self._pixels = None

    def __call__(self, state):
        from PIL import Image
        sim = self._simulator
        if self._pixels is None:
            from kinetix.render.renderer_pixels import make_render_pixels
            draw = make_render_pixels(sim.params,sim.static.replace(downscale=2))

            def render(state):
                rgb = draw(state)
                # Upstream uses x/y pixel order and a bottom-left world origin.
                return sim.jnp.clip(sim.jnp.rint(rgb),0,255).astype(sim.jnp.uint8).transpose(1,0,2)[::-1]

            self._pixels = sim.jax.jit(render)
        image = Image.fromarray(sim.np.asarray(self._pixels(state)))
        with io.BytesIO() as output:
            image.save(output,format="JPEG",quality=85)
            return output.getvalue()
