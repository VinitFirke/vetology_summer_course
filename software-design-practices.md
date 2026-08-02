YAML Frontmatter

---
name: software-design-practices
description: Helps in writing clear and concise code to solve a complex probelm using framework that is separated into 7 different sections provided below. 
---

Markdown

## 1.DESIGN WHAT YOU ARE BUILDING
- What is the software application or feature?
- Who is it intended for?
- What problem does the software solve?
- What are the main problems involved and how they are related?
- see ./REFERENCE.md - REF1.

## 2. DESIGN THE USER EXPERIENCE
- What are the main user stories? (happy flows  +  alternative flows)
- If you’re adding a new feature to an existing software application, what impact does the
feature have on the overall structure of the interface? (are there big changes in the
organization of menus, navigation, and so on?)
- see ./REFERENCE.md - REF2.

## 3. UNDERSTAND THE TECHNICAL NEEDS
- What technical details need developers to know to develop the software or new feature?
- Are there new tables to add to the database? What fields?
- How will the software technically work? Are there particular algorithms or libraries that
are important?
- What will be the overall design? Which classes are needed? What design patterns are
used to model the concepts and relationships?
- What third-party software is needed to build the software or feature?
- see ./REFERENCE.md - REF3.

## 4. IMPLEMENT TESTING AND SECURITY MEASURES
- Are there specific coverage goals for the unit tests?
- What kinds of tests are needed (unit, regression, end-to-end, and so on)?
(new feature only) Are there any potential side-effects on other areas of the application when adding this feature?
- What security checks need to be in place to allow the software to ship?
(new feature only) How does the feature impact the security of the software? Is there a need for a security audit before the feature is shipped?
- see ./REFERENCE.md - REF4.

## 5. PLAN THE WORK
- How much time will it cost to develop the software or feature?
- What are the steps and how much time does step take?
- What are the developmental milestones and in what order?
- Are there any migration scripts that need to be written?
- What are the main risk factors and are there any alternative routes to take if you find out something isn’t feasible?
- What parts are absolutely required, and what parts can optionally be done at a later stage? (i.e. the Definition of Done)
- see ./REFERENCE.md - REF5.

## 6. IDENTIFY RIPPLE EFFECTS
- What needs to be done outside of designing and implementing the feature?
- What documentation needs to be updated?
- Do you need to communicate something to existing users?
- Are there other external systems that need to be updated? For example, a payment provider, email marketing, sales system?
- see ./REFERENCE.md - REF6.

## 7. UNDERSTAND THE BROADER CONTEXT

- What are limitations of the current design?
- What are possible extensions to think about for the future?
- Any other considerations to take into account such as a budget?
- see ./REFERENCE.md - REF7.
