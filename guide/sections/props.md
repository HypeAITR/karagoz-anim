---
id: props
title: Props
read_when: a character holds, drinks from, puts down or picks up an object
departments: [continuity, layout, checker]
always_for: [continuity]
tags: [props]
---

# Props

- One holder (a character's hand) or one place at every moment (`continuity/scene<N>.yaml`).
- States move one way (a glass empties when someone drinks; it only fills if the script says so).
- **Write the prop's full description every time** — "tulip-shaped Turkish tea glass", not "glass". The model lost
  the shape when the name was shortened.
- A hand that holds nothing is said to be empty. The model keeps a prop in a hand unless told the hand is empty.
- **Picking something up or putting it down is two moves:** reach out to it, then bring it back. The moment the
  hand closes on the object (or lets go) is a key pose of its own (`holds` changes there). Out and back inside one
  transition reads as a jump back in the middle, and the gate flags it ("goes back").
- An object that lies on a table or shelf is not in a character drawing until a hand holds it: do not describe it
  in an inbetween where the hand has not reached it yet (the model draws it floating or on the floor).
