---
id: feel
title: What makes held drawings feel like animation
read_when: you plan or judge any movement and want to know why it can look like a slideshow
departments: [layout, timer, checker, orchestrator]
always_for: [layout, timer, checker]
tags: [general]
---

# What makes held drawings feel like animation

TV limited animation swaps whole drawings. A character is held in one drawing for most of a shot; only the mouth
and the eyes change. That alone looks like a slideshow. It reads as animation because of four things:

1. **Inbetweens lie on a path.** Between two key poses, each new drawing is a little further along the same arc:
   the hand travels, the head rises, the hips lift — each drawing between its two neighbours, never back, never
   aside. If one drawing breaks the path, the eye sees a jump, however many drawings there are.
2. **Spacing.** Drawings bunch up near the start and the end of a move and spread out in the middle (slow in, slow
   out). With even spacing every move looks mechanical.
3. **Holds are not dead.** A held drawing still blinks, and a pose arrives with a small settle. A hold where nothing
   changes for a long time (`hold.dead_max`) reads as a frozen frame.
4. **Parts do not stop together.** The body lands, the head arrives a frame or two later, the arms after that. The
   eyes lead: eyes → head → body.

An earlier test of this studio (one character asking "what are we waiting for, dad?") looked good with three key
poses and a handful of inbetweens, because every inbetween was described concretely and looked at before use. A
later slice with about fifty drawings failed, because the inbetweens were not on a path and nobody looked. **The
number of drawings is not what makes it work; the path is.**

Numbers in `code` come from `style.yaml`, where each value names its source. A value called a "starting value" in
these sections has no outside source; it is our first guess and will be tuned by tests.
