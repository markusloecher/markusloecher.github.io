---
layout: post
title: "A Bayesian Look at the Jason Arday Story II"
tags:
  - public-discourse
  - bayesian-reasoning
  - academic-integrity
  - critical-thinking
---


Jason Arday's death is a tragedy. Whatever one's views about the controversy surrounding him, there is nothing to celebrate in the death of a 41-year-old human being.

But I have been struck by something else in the reactions to his death.

**People seem remarkably certain about what caused it.**

Some say that Arday was hounded to death by a racist media campaign. Cambridge's chancellor, Lord Chris Smith, has described what happened as a **"racist feeding frenzy"**, while many mourners at a recent vigil explicitly blamed racist media scrutiny for his death. ([The Independent][2])

Others have a very different interpretation: that a spectacular career was beginning to unravel under scrutiny, that serious questions were being raised about his academic record and personal claims, and that the resulting exposure and shame may have contributed to his despair.

**But how could anyone possibly know which of these explanations is correct at this point?**

That is the question that interests me.

## Three possible causal stories

Consider the following very simple causal graph:

```text
Racist attacks ───────→ Psychological suffering ───────→ Suicide
                              ↑
                              │
Exposure ────────────→ Shame / despair ────────────────┘

Other interacting causes ─────────────────────────────→ Suicide
```

The first pathway is certainly plausible:

$$
\text{racist attacks}
\rightarrow
\text{psychological suffering}
\rightarrow
\text{suicide}.
$$

But so is the second:

$$
\text{exposure}
\rightarrow
\text{shame/despair}
\rightarrow
\text{suicide}.
$$

And there may be a third:

$$
\text{multiple interacting causes}
\rightarrow
\text{psychological suffering}
\rightarrow
\text{suicide}.
$$

Perhaps the reality involved elements of all three.

We simply don't know.

And that is where Bayesian reasoning becomes useful.

---

## Your posterior reveals something about your prior

Suppose the evidence currently available to us is genuinely ambiguous.

Then a person who nevertheless has an extremely strong posterior belief — *"This was clearly a racist witch hunt that caused his suicide"* — must, mathematically speaking, be bringing a strong prior to the problem.

Bayes' theorem is:

$$
P(H\mid E)
\propto
P(E\mid H)P(H).
$$

If (E), the currently available evidence, does not strongly discriminate between competing hypotheses, then a very strong posterior belief cannot have come primarily from (E).

It must be coming substantially from (P(H)): **the prior belief**.

And this is what I find so revealing about the current public discourse.

Someone who confidently attributes the suicide to a racist "feeding frenzy" is effectively assigning very little probability to the alternative causal pathways:

$$
\text{exposure}
\rightarrow
\text{shame/despair}
\rightarrow
\text{suicide}
$$

and

$$
\text{other causes}
\rightarrow
\text{suicide}.
$$

In causal-graph language, they are effectively **cutting off alternative pathways before the evidence has established that they should be cut off**.

That may turn out to be correct.

But it is not something we currently know.

---

## The same problem exists in the opposite direction

And this is important: **the Bayesian argument cuts both ways.**

Someone who says:

> "He was exposed as a fraud, became ashamed, and killed himself"

is making exactly the same epistemic mistake.

We don't know that either.

Perhaps the public scrutiny really was experienced by Arday as a racist campaign. Perhaps racism contributed substantially to his suffering. Perhaps it was decisive.

Perhaps the allegations themselves were substantially justified and the exposure contributed to his despair.

Perhaps both things happened simultaneously.

Perhaps neither captures the full story.

**A suicide does not come with a causal label attached to it.**

The fact that event A preceded event B does not establish:

$$
A \rightarrow B.
$$

And especially not:

$$
A \rightarrow B
\quad\text{and}\quad
A\text{ was racially motivated}.
$$

Those are additional causal claims requiring additional evidence.

---

# What about the Liverpool John Moores investigation?

This is another place where I think the public discussion has become too binary.

Liverpool John Moores University investigated allegations concerning Arday's PhD and did not uphold the plagiarism allegation, describing the apparent similarities as probably resulting from "honest and reasonable error". Arday subsequently emphasized that multiple investigations had concluded there was no plagiarism or academic misconduct. ([The Guardian][3])

That is certainly **evidence in his favor**.

But I would hesitate to describe it as an entirely independent exoneration.

LJMU awarded the PhD in the first place. It therefore has an obvious institutional interest in the outcome: a finding that the thesis was plagiarized would raise uncomfortable questions not only about the candidate, but also about the university's own examination and quality-control processes.

That does **not** mean that LJMU's investigation was dishonest.

It does mean that its conclusion should not automatically be treated as though it came from a completely disinterested external referee.

This is a classic Bayesian distinction:

> **Evidence can be relevant without being independent.**

And independence matters enormously when we are combining evidence.

Cambridge itself explicitly states that investigations are normally carried out by the institution where the research was undertaken, because that institution has access to the information necessary for a full investigation. ([University of Cambridge][1])

So I would regard the LJMU finding as an important piece of evidence — but **not the final word**.

---

# And this is precisely why I dislike the current polarization

There seems to be an enormous temptation to put the entire story into one of two boxes:

**BOX A**

> Jason Arday was an innocent Black academic subjected to a racist witch hunt, and this caused his suicide.

or:

**BOX B**

> Jason Arday was a fraud whose extraordinary story was finally exposed, and the shame caused his suicide.

But why should reality have to fit either box?

There are many possible combinations:

$$
\text{legitimate scrutiny}
+
\text{racist abuse}
+
\text{personal vulnerability}
+
\text{professional shame}
+
\text{other personal factors}
\rightarrow
\text{suicide}.
$$

Some of these variables could be important.

Some could be irrelevant.

Some could turn out to be completely wrong.

**We don't know yet.**

And if we don't know, then the appropriate intellectual response is not certainty.

It is **updating**.

---

# The uncomfortable conclusion

I suspect that the strongest reactions to this story tell us at least as much about the people making them as they do about Jason Arday.

If someone sees an ambiguous set of facts and immediately concludes:

> *"This is obviously another racist witch hunt,"*

I want to know what prior beliefs about race, academia and discrimination they brought to the table.

And if someone sees the same facts and immediately concludes:

> *"This is obviously a fraud who was finally exposed,"*

I want to know what prior beliefs they brought to the table too.

**Both may ultimately turn out to be right.**

But neither conclusion is currently established merely by the fact that Arday died.

That is precisely why I think we should resist the urge to turn this tragedy into confirmation of whatever worldview we already held.

**The evidence should determine the posterior.**

Not the other way around.

And perhaps the most honest position at the moment is also the least satisfying:

> **I don't know what happened. I don't know what caused his death. And I am willing to change my mind as we find out more.**

That isn't indecision.

**It is what updating one's beliefs is supposed to look like.**

[1]: https://www.cam.ac.uk/notices/news/statement-to-media-02-august-2026?utm_source=chatgpt.com "Statement to media 02 August 2026 | University of Cambridge"
[2]: https://www.independent.co.uk/bulletin/news/jason-arday-death-lord-chris-smith-b3034191.html?utm_source=chatgpt.com "Jason Arday faced ‘racist feeding frenzy’, says Cambridge University’s chancellor"
[3]: https://www.theguardian.com/education/2026/aug/05/cambridge-professor-jason-arday-resigns-amid-accusations-of-plagiarism?utm_source=chatgpt.com "Cambridge professor Jason Arday resigns amid accusations of plagiarism | University of Cambridge | The Guardian"
