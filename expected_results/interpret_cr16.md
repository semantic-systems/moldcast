# What the slots of cr16 learn (VitaminC test, 3,932 queries)

Edit types: number 2,386, name 841, other 669, negation 36

## Summary per seed

| Seed | Pairwise | Slots used | Slots for 80% of a clash | Top slot's share | Edit-type NMI (shuffled; p) | Attention on the edit: clash slot / other slots / chance |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 41.2% | 16 of 16 | 13.0 | 7% | 0.004 (0.004; p = 0.368) | 11% / 11% / 11% |
| 1 | 39.2% | 16 of 16 | 13.0 | 7% | 0.003 (0.004; p = 0.720) | 11% / 11% / 11% |
| 2 | 39.3% | 16 of 16 | 13.0 | 7% | 0.004 (0.004; p = 0.451) | 11% / 11% / 11% |

## Seed 0: slot by slot

Dominant: share of queries whose clash this slot carries most. Ablation: drop in pairwise accuracy (points) when the slot is switched off, overall and per edit type.

| Slot | Commits | Dominant | Its main edit type | Ablation: overall | number | negation | name | other | Attention on the edit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 100% | 7% | number (63%) | 1.9 | 2.7 | 0.0 | 0.8 | 0.3 | 10% |
| 1 | 100% | 6% | number (57%) | 1.9 | 2.6 | 0.0 | 1.0 | 0.3 | 10% |
| 2 | 100% | 5% | number (68%) | 1.9 | 2.7 | 0.0 | 1.0 | 0.3 | 11% |
| 3 | 100% | 7% | number (65%) | 1.9 | 2.7 | 0.0 | 1.0 | 0.3 | 11% |
| 4 | 100% | 6% | number (60%) | 1.9 | 2.6 | 0.0 | 1.0 | 0.3 | 12% |
| 5 | 100% | 6% | number (54%) | 1.9 | 2.7 | 0.0 | 0.8 | 0.3 | 10% |
| 6 | 100% | 6% | number (57%) | 1.9 | 2.7 | 0.0 | 1.0 | 0.3 | 11% |
| 7 | 100% | 6% | number (58%) | 1.9 | 2.7 | 0.0 | 0.8 | 0.3 | 10% |
| 8 | 100% | 8% | number (61%) | 1.9 | 2.6 | 0.0 | 1.0 | 0.3 | 11% |
| 9 | 100% | 7% | number (59%) | 1.9 | 2.7 | 0.0 | 1.0 | 0.3 | 10% |
| 10 | 100% | 7% | number (59%) | 1.9 | 2.7 | 0.0 | 0.8 | 0.3 | 12% |
| 11 | 100% | 6% | number (63%) | 1.9 | 2.7 | 0.0 | 0.8 | 0.3 | 10% |
| 12 | 100% | 5% | number (61%) | 1.9 | 2.7 | 0.0 | 1.0 | 0.3 | 11% |
| 13 | 100% | 5% | number (66%) | 1.9 | 2.7 | 0.0 | 1.0 | 0.3 | 10% |
| 14 | 100% | 8% | number (59%) | 1.9 | 2.7 | 0.0 | 1.0 | 0.3 | 10% |
| 15 | 100% | 6% | number (60%) | 1.8 | 2.6 | 0.0 | 1.0 | 0.3 | 12% |

Most important slots (largest ablation drop), with pairs whose clash they carry:

Slot 2:
- number: *300 -> 400* (6% of the clash). Target: The capital city , Muscat continues to be the region with the highest number of confirmed cases at over 400 .
- other: *generally -> mixed to* (6% of the clash). Target: The Wedding Ringer received mixed to negative reviews from critics .
- number: *2078 -> 2079* (6% of the clash). Target: In 2079 , CIA agent Snow is arrested for murdering undercover agent Frank Armstrong , who had uncovered evidence of another agent selling secrets about the space program .
- name: *North South -> Republic of North* (6% of the clash). Target: It is bordered by Japan , Republic of Korea , Russia and North Korea .

Slot 3:
- number: *71 109 -> 70 108* (6% of the clash). Target: A Madea Halloween and Geostorm. On review aggregator website Rotten Tomatoes , the film has an approval rating of 70 % based on 108 reviews , with an average rating of 6/10 .
- number: *67 4.3 -> 66 4.2* (6% of the clash). Target: On Rotten Tomatoes , the film has a rating of 23 % , based on 66 reviews , with an average rating of 4.2/10 .
- number: *10.19 -> 10.5* (6% of the clash). Target: Northwestern has the eighth largest university endowment in the United States , currently valued at $ 10.5 billion .
- number: *89 -> 96* (6% of the clash). Target: The Big Wedding holds a 7 % rating on Rotten Tomatoes based on 96 reviews with the site 's consensus stating that its `` all-star cast is stranded in a contrived , strained plot that features broad stabs at humor but few laughs .

Slot 6:
- other: *positive -> mixed* (6% of the clash). Target: Speak Your Mind received mixed reviews from music critics .
- number: *62 -> 60* (6% of the clash). Target: Its lead singles `` Liar '' and `` Shameless '' peaked at number 52 and 60 on the Billboard Hot 100 , respectively.
- other: *positive -> negative* (6% of the clash). Target: Thugs of Hindostan received generally negative reviews from critics .
- number: *46 -> 45* (6% of the clash). Target: As of 27 February , there have been 45 confirmed cases and 16 recoveries.

## Seed 1: slot by slot

Dominant: share of queries whose clash this slot carries most. Ablation: drop in pairwise accuracy (points) when the slot is switched off, overall and per edit type.

| Slot | Commits | Dominant | Its main edit type | Ablation: overall | number | negation | name | other | Attention on the edit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 100% | 6% | number (59%) | 2.1 | 2.3 | 0.0 | 1.4 | 2.1 | 11% |
| 1 | 100% | 6% | number (64%) | 2.0 | 2.3 | 0.0 | 1.4 | 2.1 | 9% |
| 2 | 100% | 7% | number (63%) | 2.1 | 2.3 | 0.0 | 1.4 | 2.1 | 10% |
| 3 | 100% | 5% | number (64%) | 2.0 | 2.3 | 0.0 | 1.4 | 2.1 | 10% |
| 4 | 100% | 6% | number (62%) | 2.1 | 2.3 | 0.0 | 1.4 | 2.1 | 9% |
| 5 | 100% | 6% | number (61%) | 2.1 | 2.3 | 0.0 | 1.4 | 2.1 | 12% |
| 6 | 100% | 7% | number (59%) | 2.1 | 2.3 | 0.0 | 1.4 | 2.1 | 10% |
| 7 | 100% | 6% | number (60%) | 2.1 | 2.3 | 0.0 | 1.4 | 2.1 | 11% |
| 8 | 100% | 6% | number (60%) | 2.1 | 2.3 | 2.8 | 1.4 | 2.1 | 11% |
| 9 | 100% | 7% | number (66%) | 2.1 | 2.3 | 0.0 | 1.4 | 2.1 | 11% |
| 10 | 100% | 6% | number (58%) | 2.1 | 2.3 | 0.0 | 1.4 | 2.1 | 10% |
| 11 | 100% | 7% | number (55%) | 2.1 | 2.3 | 0.0 | 1.4 | 2.1 | 10% |
| 12 | 100% | 7% | number (56%) | 2.1 | 2.3 | 0.0 | 1.4 | 2.1 | 12% |
| 13 | 100% | 7% | number (58%) | 2.1 | 2.3 | 0.0 | 1.4 | 2.1 | 11% |
| 14 | 100% | 6% | number (60%) | 2.1 | 2.3 | 0.0 | 1.4 | 2.1 | 10% |
| 15 | 100% | 6% | number (65%) | 2.1 | 2.3 | 2.8 | 1.4 | 2.1 | 12% |

Most important slots (largest ablation drop), with pairs whose clash they carry:

Slot 8:
- number: *0.8 0.9 -> 1.14 0.38* (6% of the clash). Target: Islam ( 1.14 % ) and Judaism ( 0.38 % ) are also practiced .
- number: *85 -> 87* (6% of the clash). Target: On 4 March the total number of confirmed cases increased to 87 .
- number: *30 -> 26* (6% of the clash). Target: `` On Metacritic , the film has a score of 68 out of 100 , based on 26 reviews , indicating `` generally favorable reviews '' .
- number: *45 -> 46* (6% of the clash). Target: As of 27 February , there have been 46 confirmed cases and 16 recoveries.

Slot 15:
- other: *six -> five* (6% of the clash). Target: It also became Dion 's first out of five Diamond-certified albums in France. < ref name= '' snep '' > French certifications : * For À l'Olympia , Les premières chansons vol .
- number: *62 -> 60* (6% of the clash). Target: Its lead singles `` Liar '' and `` Shameless '' peaked at number 52 and 60 on the Billboard Hot 100 , respectively.
- number: *142,940,100 . -> 142* (7% of the clash). Target: Taking in an additional $ 98,321,000 , the movie grossed a worldwide total of $ 142
- number: *312 -> 32* (6% of the clash). Target: On its first weekend , the total collection was noted as 32 crore .

Slot 0:
- other: *three -> eight* (6% of the clash). Target: O'Leary was born in Stoke Newington , London on 2 May 1958 and moved to live in Dublin at the age of eight.
- number: *2,900 -> 3,000* (6% of the clash). Target: Collateral Beauty will open alongside Rogue One and is expected to gross $ 10–13 million from 3,000 theaters in its opening weekend.
- number: *158.1 292.3 -> 157.7 291.8* (6% of the clash). Target: , John Wick : Chapter 3 – Parabellum has grossed $ 157.7 million in the United States and Canada , and $ 134.1 million in other territories , for a worldwide total of $ 291.8 million .
- number: *457 -> 449* (6% of the clash). Target: As of 1 April 2020 , there are 449 confirmed cases in 23 out of 28 Bulgarian provinces , with 10 recorded deaths and 25 recoveries .

## Seed 2: slot by slot

Dominant: share of queries whose clash this slot carries most. Ablation: drop in pairwise accuracy (points) when the slot is switched off, overall and per edit type.

| Slot | Commits | Dominant | Its main edit type | Ablation: overall | number | negation | name | other | Attention on the edit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 100% | 6% | number (63%) | 1.9 | 2.3 | 2.8 | 1.1 | 1.5 | 11% |
| 1 | 100% | 6% | number (56%) | 1.9 | 2.3 | 2.8 | 1.1 | 1.5 | 10% |
| 2 | 100% | 6% | number (63%) | 1.9 | 2.3 | 2.8 | 1.1 | 1.5 | 11% |
| 3 | 100% | 7% | number (59%) | 1.9 | 2.3 | 2.8 | 1.1 | 1.5 | 12% |
| 4 | 100% | 7% | number (67%) | 1.9 | 2.3 | 2.8 | 1.1 | 1.5 | 10% |
| 5 | 100% | 6% | number (63%) | 1.9 | 2.3 | 2.8 | 1.1 | 1.5 | 11% |
| 6 | 100% | 7% | number (62%) | 1.9 | 2.3 | 2.8 | 1.1 | 1.5 | 10% |
| 7 | 100% | 6% | number (62%) | 1.9 | 2.3 | 2.8 | 1.1 | 1.5 | 9% |
| 8 | 100% | 6% | number (53%) | 1.9 | 2.3 | 2.8 | 1.1 | 1.5 | 11% |
| 9 | 100% | 6% | number (65%) | 1.9 | 2.3 | 2.8 | 1.1 | 1.5 | 10% |
| 10 | 100% | 6% | number (60%) | 2.0 | 2.4 | 2.8 | 1.1 | 1.5 | 12% |
| 11 | 100% | 7% | number (58%) | 1.9 | 2.3 | 2.8 | 1.1 | 1.5 | 11% |
| 12 | 100% | 7% | number (62%) | 1.9 | 2.3 | 2.8 | 1.1 | 1.5 | 10% |
| 13 | 100% | 6% | number (57%) | 1.9 | 2.3 | 2.8 | 1.1 | 1.5 | 11% |
| 14 | 100% | 7% | number (59%) | 1.9 | 2.3 | 2.8 | 1.1 | 1.5 | 11% |
| 15 | 100% | 5% | number (64%) | 1.9 | 2.3 | 2.8 | 1.1 | 1.5 | 12% |

Most important slots (largest ablation drop), with pairs whose clash they carry:

Slot 10:
- number: *November 2019 -> March 2020* (6% of the clash). Target: The Jonas Brothers ' joint memoir , Blood ( co-written with Neil Strauss ) publishes in March 2020.
- number: *March 2020 -> November 2019* (6% of the clash). Target: The Jonas Brothers ' joint memoir , Blood ( co-written with Neil Strauss ) publishes in November 2019.
- number: *10 890 January 17 -> 11 949 November 2* (6% of the clash). Target: This six-minute long video , titled `` Coming Out '' , has over 11 million views and over 949 thousand likes , being the second most viewed video on Franta 's channel as of November 2 , 2016.
- number: *2078 -> 2079* (6% of the clash). Target: In 2079 , CIA agent Snow is arrested for murdering undercover agent Frank Armstrong , who had uncovered evidence of another agent selling secrets about the space program .

Slot 9:
- other: *three -> four* (6% of the clash). Target: He was also a director for two episodes in the first season , `` Baelor '' and `` Fire and Blood '' , as well as four other episodes in the second season .
- number: *11 949 November 2 -> 10 890 January 17* (6% of the clash). Target: This six-minute long video , titled `` Coming Out '' , has over 10 million views and over 890 thousand likes , being the second most viewed video on Franta 's channel as of January 17 , 2016.
- number: *26 -> 30* (6% of the clash). Target: `` On Metacritic , the film has a score of 68 out of 100 , based on 30 reviews , indicating `` generally favorable reviews '' .
- number: *800 -> 810* (6% of the clash). Target: The music video has garnered over 810 million views on YouTube .

Slot 14:
- other: *three -> eight* (6% of the clash). Target: O'Leary was born in Stoke Newington , London on 2 May 1958 and moved to live in Dublin at the age of eight.
- number: *71 109 -> 70 108* (6% of the clash). Target: A Madea Halloween and Geostorm. On review aggregator website Rotten Tomatoes , the film has an approval rating of 70 % based on 108 reviews , with an average rating of 6/10 .
- number: *45 -> 46* (6% of the clash). Target: As of 27 February , there have been 46 confirmed cases and 16 recoveries.
- number: *6 104 -> 7 105* (6% of the clash). Target: Rotten Tomatoes , a review aggregator , reports that 7 % of 105 critics have given Paranoia a positive review ; the average rating is 3.8/10 .
