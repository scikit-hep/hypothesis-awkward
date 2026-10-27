document$.subscribe(function () {
  document.querySelectorAll('a[href^="http"]').forEach(a => {
    if (!a.href.startsWith(window.location.origin)) {
      a.setAttribute("target", "_blank");
      a.setAttribute("rel", "noopener");
    }
  });
});
