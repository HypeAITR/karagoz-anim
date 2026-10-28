---
id: handover
title: Handing a prop from one character to another
read_when: a prop goes from one character's hand to another's (giving, taking, passing)
departments: [continuity, layout, timer, checker]
always_for: [continuity]
tags: [handover, props]
---

# Handing a prop from one character to another

A prop has exactly one holder at every moment, except **the moment of contact**, when both hands are on the same
object. Two characters are drawn separately, so nothing joins their hands except the plan:

1. Before: the giver holds the prop out; the receiver reaches with an open, empty hand.
2. **Contact: both hands on the same prop.** Make this a key pose for both characters, landing on the same anchor
   and held at least `hold.min` frames. In the giver's drawing the prop is in their hand; in the receiver's drawing
   their hand is closed around a prop at the same place. The engine shows them together.
3. After: the giver's hand opens and is empty; the receiver draws the prop in.

Never let the prop move to another place in the giver's drawing (to the other hand, the lap) during the handover,
and never let the receiver hold it before contact: the result is two props on screen.

Who does what: Continuity says on which word or moment the hands meet and who holds the prop before and after.
Layout gives both characters the contact key pose and states the hands in every inbetween (in the sentence and in
the transition's `holds`). Timer lands both contact
poses on the same anchor with the same cadence and holds them. Checkers look at the rendered frames where both are
together.

Why this section exists: in an early slice the handover was spread over inbetweens of both characters. In one
drawing the giver moved the glass to the lap while the reaching hand was empty; in the next both characters held a
glass. Two glasses were on screen for two frames.
