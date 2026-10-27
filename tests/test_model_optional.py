"""Actual pretrained upstream RTC smoke test, enabled explicitly with RTC_CHECKPOINT."""
import os
import unittest


@unittest.skipUnless(os.environ.get("RTC_CHECKPOINT"),"Set RTC_CHECKPOINT after installing RTC dependencies")
class PretrainedTests(unittest.TestCase):
    def test_actual_checkpoint_generates_finite_chunks(self):
        from wireless_rtc.simulation.kinetix import KinetixPolicy
        p=KinetixPolicy(os.environ["RTC_CHECKPOINT"])
        jax,jnp,np=p.jax,p.jnp,p.np
        obs,state=p.reset(jax.random.key(0))
        next_obs,_,_,_,_=p.step(jax.random.key(3),state,jnp.zeros(p.action_dim))
        self.assertTrue(np.isfinite(np.asarray(next_obs)).all())
        previous=p.plain(jax.random.key(1),obs[None])
        for d in [0,2,4]:
            new=p.guided(jax.random.key(2),obs[None],previous,d,4)
            self.assertEqual(new.shape,previous.shape)
            self.assertTrue(np.isfinite(np.asarray(new)).all())


if __name__=="__main__":
    unittest.main()
