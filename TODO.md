# NCA Dino Game - TODO List

## High Priority
- [ ] **Improve mobile scaling** - Current approach doesn't work properly on mobile devices
  - Game area (768px wide) doesn't fit on mobile screens
  - Need responsive solution that maintains game mechanics
  - Previous attempts with CSS transform/zoom failed

## Medium Priority  
- [ ] **Explore why Firefox is slower** - Firefox shows 2.5x slower performance
  - Other browsers: ~17ms per NCA step
  - Firefox: ~40-45ms per NCA step  
  - May be related to WASM provider vs CPU provider
  - Warning seen: [Firefox WASM source map error]

- [ ] **Add bomb button to mobile interface** - Mobile users need touch controls for bombs
  - Current controls: SPACEBAR = jump, B = bomb
  - Mobile has touch for jump, but no bomb control
  - Need visible bomb button for mobile users

## Completed ✅
- [x] Adaptive sparse optimization (2.7x speedup achieved)
- [x] GitHub Pages deployment 
- [x] Service worker for SharedArrayBuffer support
- [x] Variable gravity jump physics
- [x] Basic mobile touch controls for jumping

---

*Last updated: 2025-01-06*