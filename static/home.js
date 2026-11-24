// Pagina iniziale: comparsa delle fasi durante lo scorrimento e barra in alto.

const stepNav = document.getElementById('stepNav');
const hero = document.getElementById('hero');
const progressBar = document.getElementById('progressBar');
const sections = [...document.querySelectorAll('.step')];
const links = [...document.querySelectorAll('[data-step]')]; // barra in alto e pallini
const dotsFill = document.getElementById('dotsFill');

// Senza questa classe il contenuto è già tutto visibile
document.documentElement.classList.add('animate');

// I blocchi compaiono entrando dal basso, e si ripetono se si torna su
const reveal = new IntersectionObserver((entries) => {
    for (const entry of entries) {
        if (entry.isIntersecting) {
            entry.target.classList.add('shown');
        } else if (entry.boundingClientRect.top > window.innerHeight * 0.8) {
            entry.target.classList.remove('shown');
        }
    }
}, { threshold: 0.15, rootMargin: '0px 0px -8% 0px' });
document.querySelectorAll('.reveal').forEach((node) => reveal.observe(node));

let scheduled = false;
function update() {
    scheduled = false;
    stepNav.classList.toggle('visible', hero.getBoundingClientRect().bottom < window.innerHeight * 0.45);

    // Fase corrente: l'ultima che ha superato la metà dello schermo
    const middle = window.innerHeight / 2;
    let current = null;
    for (const section of sections) {
        if (section.getBoundingClientRect().top <= middle) current = section;
    }
    const index = sections.indexOf(current);
    for (const link of links) {
        const position = sections.findIndex((section) => section.id === link.dataset.step);
        if (position === index) link.setAttribute('aria-current', 'step');
        else link.removeAttribute('aria-current');
        link.classList.toggle('passed', position < index);
    }
    dotsFill.style.transform = `scaleY(${Math.max(0, index) / (sections.length - 1)})`;

    const start = sections[0].getBoundingClientRect().top;
    const end = sections[sections.length - 1].getBoundingClientRect().bottom;
    const done = Math.min(1, Math.max(0, (middle - start) / (end - start)));
    progressBar.style.transform = `scaleX(${done})`;
}

function schedule() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(update);
}

window.addEventListener('scroll', schedule, { passive: true });
window.addEventListener('resize', schedule);
update();
