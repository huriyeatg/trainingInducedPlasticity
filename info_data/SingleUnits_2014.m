% single/multiple data for each penetration
animalList = [0832 0833 0848 0943 0963];
penetrationList = [10,11,12,1:9,1:6,1:10,12:19,1:21];

S_units = [%animal, site, shrankNo,[SingleUnitNo-1sh],,[SingleUnitNo-1sh],...
    %[SingleUnitNo-2sh],[SingleUnitNo-3sh],[SingleUnitNo-4sh]];
{0832 10,[],[],NaN,NaN}%
{0832 11,[],[],NaN,NaN}%
{0832 12,[],[1,4],NaN,NaN}%
{0833 1,[],NaN,NaN,NaN}%
{0833 2,[3,5,8,9,11,12],[1,4,5,7,8],NaN,NaN}%
{0833 3,[],[1,2,3],NaN,NaN}%
{0833 4,[2:5],[2,3,4,6],NaN,NaN}%
{0833 5,[3],[2:8],NaN,NaN}%
{0833 6,[2,3,6,8],[2,6,7],NaN,NaN}%
{0833 7,[2:4,6:9],[2:5],NaN,NaN}%
{0833 8,[],[],NaN,NaN}%
{0833 9,[],[],NaN,NaN}%
{0848 1,[3],[1],NaN,NaN}%
{0848 2,[6],[2],NaN,NaN}%
{0848 3,[2:5],[5],NaN,NaN}%
{0848 4,[9:11],[2,6],NaN,NaN}%
{0848 5,[2,3],[5,6,8],NaN,NaN}%
{0848 6,[],[3],NaN,NaN};%
{0943 1,[1,3,4,6,7,8],[1:4,7:12],[1:5,7,8,10:13],[1,3:7,10,13]}%
{0943 2,[2,3,4,8,9],[6,9],NaN,NaN}%
{0943 3,[],[2,4,6,8],NaN,NaN}%
{0943 4,[4:7],[5],NaN,NaN}%
{0943 5,[],[3:7],NaN,NaN}%
{0943 6,[2,3],[3,4,5,7,8,10],NaN,NaN}%
{0943 7,[6],[8,9,11,12,14],[7,9,10],[8:10]}%
{0943 8,[2],[7],NaN,NaN}%
{0943 9,[2,3,5,6],[2,7],NaN,NaN}%
{0943 10,[1:8],[2:12],NaN,NaN}%
{0943 12,[3,5,7,8],[2,3,7,8],NaN,NaN}%
{0943 13,[1,2,3,4,5],[],NaN,NaN}%
{0943 14,[1,2],[3:6],NaN,NaN}%
{0943 15,NaN,[8,9],NaN,NaN}%
{0943 16,[1],[3,4,5,10,11],NaN,NaN}%
{0943 17,[],[1:6],NaN,NaN}%
{0943 18,[],[1:4,6:10],NaN,NaN}%
{0943 19,[2],[2:9],NaN,NaN}%
{0963 1,[3,8],[3],NaN,NaN}%
{0963 2,[2:5,8],[3:5],NaN,NaN}%
{0963 3,[1,3,4,6,9],[3],NaN,NaN}%
{0963 4,[3,4,6,7],[4:6],NaN,NaN}%
{0963 5,[2,5],[2,4,5],NaN,NaN}%
{0963 6,[2,3,4,7],[],NaN,NaN}%
{0963 7,[3,4,5,7],[3,4],NaN,NaN}%
{0963 8,[],[2,3,5],NaN,NaN}%
{0963 9,[2:10],[1,3,5],NaN,NaN}%
{0963 10,[],[],NaN,NaN}%
{0963 11,[3:9],[3,4,5,8],NaN,NaN}%
{0963 12,[1:10],[1:5,8:12],NaN,NaN}%
{0963 13,[2:4,6:9],[3:9,12:14],NaN,NaN}%
{0963 14,[2:8],[2,6,10],NaN,NaN}%
{0963 15,[1:7],[2,5,7,8,9],NaN,NaN}%
{0963 16,[2,3,5:9],[3:7,9],NaN,NaN}%
{0963 17,[2:7],[1:4],NaN,NaN}%
{0963 18,[5:8],[2:4],NaN,NaN}%
{0963 19,NaN,[2,3,4],NaN,NaN}%
{0963 20,[1:9],[3:8],NaN,NaN}%
{0963 21,[],[3,4,5,6],NaN,NaN}%
];
