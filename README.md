# CardioPredict AI

Heart disease risk predictor using logistic regression trained on the Cleveland Heart Disease Dataset (UCI).

## Quick start

```bash
npm install
npm start
```

## Deploy to Vercel

```bash
npx vercel
```

## Deploy to GitHub Pages

1. Add to package.json: `"homepage": "https://YOUR_USERNAME.github.io/heart-predictor"`
2. Install: `npm install gh-pages --save-dev`
3. Add scripts: `"predeploy": "npm run build"` and `"deploy": "gh-pages -d build"`
4. Run: `npm run deploy`

## Tech

- React 18
- Logistic regression (pure JS, no ML library needed)
- 100 real patient records from the Cleveland Heart Disease Dataset
- lucide-react icons
