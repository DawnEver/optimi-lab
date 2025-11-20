# **Unit/Integration/E2E Testing**
---

### 1. **Unit Testing (Unit Testing)**
- **Definition**: Tests for the smallest testable units of code (e.g., functions, class methods).
- **Scope**: Isolated tests for single components, **mock/stub all external dependencies** (e.g., databases, APIs, other classes).
- **Purpose**: Verify correctness of code logic and ensure each unit works as expected.

---

### 2. **Integration Testing (Integration Testing)**
- **Definition**: Tests whether interactions between multiple components or services work correctly.
- **Scope**: Verify integration between modules, services, or subsystems, **use real dependencies partially** (e.g., APIs).
- **Purpose**: Find interface errors, data flow issues, or dependency compatibility problems.

---

### 3. **E2E Testing (End-to-End Testing)**
- **Definition**: Simulate real user scenarios, testing the complete flow from the user interface to backend systems.
- **Scope**: Cover the entire application stack (UI, API, external services), **use all real dependencies**.
- **Purpose**: Verify whether the system meets user requirements and find cross-layer issues (e.g., front-end to back-end data inconsistencies).

---

### **Key Comparisons of the Three**
| Dimension           | Unit Testing          | Integration Testing       | E2E Testing               |
|--------------------|-----------------------|---------------------------|---------------------------|
| **Test Scope**      | Single function/class | Inter-module interactions | Full user flows           |
| **External Dependencies** | All mocked/stubbed    | Part real, part mocked     | All real (or production images) |
| **Execution Speed** | Milliseconds          | Seconds                   | Minutes                   |
| **Types of Issues Found** | Logic errors           | Interface compatibility, data flow issues | Cross-system interaction, user experience issues |

---

### **How to Choose?**
- **Unit Testing**: Ensure baseline code quality, suitable for TDD (Test-Driven Development).
- **Integration Testing**: Verify the collaboration of key modules.
- **E2E Testing**: Cover core user scenarios.

**Best Practices**:
Adopt the **testing pyramid** model (most unit tests, fewest E2E), for example:
- 70% Unit tests
- 20% Integration tests
- 10% E2E tests

By layered testing, you can maximize efficiency and maintainability while ensuring quality.
